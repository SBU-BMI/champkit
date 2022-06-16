"""Save the images in the HDF5 files as individual PNG files."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def process_one_partition(data_path: Path, partition: str):
    print(f"[champkit] working on PatchCamelyon {partition} split")

    root = Path("images")

    if (root / partition).exists():
        print("[champkit] skipping: partition already created")
        return

    xpath = data_path / f"camelyonpatch_level_2_split_{partition}_x.h5"
    ypath = data_path / f"camelyonpatch_level_2_split_{partition}_y.h5"

    with h5py.File(xpath, "r") as f:
        x = f["x"][()]
    print("[champkit] X.shape", x.shape)

    with h5py.File(ypath, "r") as f:
        y = f["y"][()]
    y = y.squeeze()
    assert np.array_equal(np.unique(y), [0, 1])
    print("[champkit] y.shape", y.shape)

    assert x.shape[0] == y.shape[0], "different num of x and y samples"

    (root / partition / "tumor-negative").mkdir(parents=True, exist_ok=True)
    (root / partition / "tumor-positive").mkdir(parents=True, exist_ok=True)

    print("[champkit] Saving images...")
    for j in range(x.shape[0]):
        this_label = y[j].item()
        if this_label == 0:
            this_label = "tumor-negative"
        elif this_label == 1:
            this_label = "tumor-positive"
        else:
            raise ValueError("unknown label")
        Image.fromarray(x[j], mode="RGB").save(root / partition / this_label / f"{j:06d}.png")


if __name__ == "__main__":

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_path", help="Path containing the .h5 files")
    args = p.parse_args()
    args.data_path = Path(args.data_path)

    process_one_partition(data_path=args.data_path, partition="valid")
    process_one_partition(data_path=args.data_path, partition="test")
    process_one_partition(data_path=args.data_path, partition="train")
