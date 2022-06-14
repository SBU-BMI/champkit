"""Split trainval data from Kather et al. 2019 into train and val partitions."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_df_with_splits_msi_mss(trainval_dir: str) -> pd.DataFrame:
    root = Path(trainval_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    paths_msimut = list((root / "MSIMUT").glob("*.png"))
    paths_mss = list((root / "MSS").glob("*.png"))
    assert paths_msimut and paths_mss
    # Make sure there MSI and MSS labels do not share images.
    assert not set(p.name for p in paths_msimut).intersection(p.name for p in paths_mss)

    df = pd.DataFrame({"MSIMUT": paths_msimut, "MSS": paths_mss})
    df = df.melt(var_name="label", value_name="path").loc[:, ["path", "label"]]

    df_train, df_val = train_test_split(
        df, stratify=df.label.values, train_size=0.8, random_state=42
    )
    df_train.loc[:, "partition"] = "train"
    df_val.loc[:, "partition"] = "val"
    df = pd.concat((df_train, df_val)).reset_index(drop=True)
    assert not df.duplicated().any(), "found duplicate rows"
    return df


def main_msi_mss(trainval_dir: str) -> pd.DataFrame:
    assert Path("images").exists(), "expected 'images' directory to exist"
    df = get_df_with_splits_msi_mss(trainval_dir)
    df = df.reset_index(drop=True)

    print(f"[champkit] Found {len(df):,} MSI/MSS samples for this in {trainval_dir}")

    def make_symlink(row):
        src = Path(row["path"])
        label = row["label"]
        partition = row["partition"]
        link = Path(f"images/{partition}/{label}/{src.name}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(f"../../trainval/{label}/{src.name}")
        return link

    print("[champkit] Splitting trainval directory into train/val partitions ...")
    df.loc[:, "symlink"] = df.apply(make_symlink, axis=1)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Partition data from Kather et al. 2019")
    p.add_argument("trainval_dir")
    args = p.parse_args()
    df = main_msi_mss(trainval_dir=args.trainval_dir)
    print(f"[champkit] Saving partition info to 'trainval-partitions.csv'")
    df.to_csv("trainval-partitions.csv", index=False)
