#!/usr/bin/env bash
#
# Download and prepare the PatchCamelyon dataset.

set -eu

echo
echo
echo "[champkit] ----------------- SETUP TASK 1 -----------------"

# Get directory of current script.
here="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# Exit early if dataset seems to be ready.
if [[ -d $here/images ]] && [[ -f $here/images/classmap.txt ]]; then
    echo "[champkit] found $here/images directory... not doing anything because this dataset seems to be ready."
    exit 0
fi

download_dir=downloaded-h5

# All paths in this script are relative to the script's directory.
cd $here

# Download ---------------------------------------------------
mkdir -p $download_dir
cd $download_dir

echo "[champkit] Downloading data to $here"
echo "[champkit] This may take a while..."

wget_args="--no-clobber --content-disposition --quiet --show-progress"

# Download test set.
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz?download=1
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz?download=1

# Download training set.
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz?download=1
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz?download=1

# Download validation set.
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz?download=1
wget $wget_args https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz?download=1

echo "[champkit] Verifying integrity of downloaded files..."
md5sum --check ../md5sums.txt

echo "[champkit] Uncompressing the .h5.gz files"
echo "[champkit] This may take a while..."
find . -name '*.h5.gz' -exec gunzip {} \;

echo "[champkit] Moving images from HDF5 files to individual PNG files..."
cd ..
python save_images.py $download_dir

# To keep things consistent, all tasks are assumed to have train/val/test directories.
mv images/valid images/val

echo "[champkit] Removing downloaded files (they are no longer needed)..."
rm -r $download_dir

echo "[champkit] Generating classmap.txt"
# save_images.py creates the images directory.
printf "tumor-negative\ntumor-positive\n" > images/classmap.txt

echo "[champkit] Done!"
