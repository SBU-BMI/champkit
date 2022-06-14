"""Split MHIST data into train/val/test."""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_df_with_splits() -> pd.DataFrame:
    if not Path("annotations.csv").exists():
        raise FileNotFoundError("cannot find annotations.csv")
    # Originally the data is only split into train/test.
    df = pd.read_csv("annotations.csv")
    # Split training set into train/val.
    df_train = df.query("Partition=='train'")
    df_test = df.query("Partition=='test'")
    df_train, df_val = train_test_split(df_train, train_size=0.8, random_state=42)
    df_val.loc[:, "Partition"] = "val"
    # Re-combine train/val/test.
    df = pd.concat((df_train, df_val, df_test))
    assert not df.duplicated().any(), "found duplicate rows"
    return df


def main():
    assert Path("images").exists(), "expected 'images' directory to exist"
    root = Path("images-split")
    root.mkdir(exist_ok=True)

    def make_symlink(row):
        src = Path("images") / row["Image Name"]
        label = row["Majority Vote Label"]
        partition = row["Partition"]
        link: Path = root / partition / label / src.name
        link.parent.mkdir(exist_ok=True, parents=True)
        src = Path(os.path.relpath(src, link.parent))
        link.symlink_to(src)

    df = get_df_with_splits()
    print("[champkit] Number of images in each data split:")
    for k, v in df.Partition.value_counts().iteritems():
        print(f"[champkit]    {k}\t{v:,}")
    df.apply(make_symlink, axis=1)
    print(f"[champkit] Saved splits to {root.absolute()}")


if __name__ == "__main__":
    main()
