#!/usr/bin/env bash
#
# Download and prepare data for TIL detection task.

set -eu

echo
echo
echo "[champkit] ----------------- SETUP TASK 2 -----------------"

# Get directory of current script.
here="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# Exit early if dataset seems to be ready.
if [[ -d $here/images-tcga-tils/pancancer ]] && [[ -f $here/images-tcga-tils/pancancer/classmap.txt ]]; then
    echo "[champkit] found $here/images-tcga-tils/pancancer directory... not doing anything because this dataset seems to be ready."
    exit 0
fi

# All paths in this script are relative to the script's directory.
cd $here

# Download ---------------------------------------------------
echo "[champkit] Downloading data to $here"
echo "[champkit] This may take a while..."

wget_args="--no-clobber --content-disposition --quiet --show-progress"
wget $wget_args https://zenodo.org/record/6604094/files/TCGA-TILs.tar.gz?download=1

echo "[champkit] Verifying integrity of downloaded file..."
md5sum --check md5sums.txt

echo "[champkit] Extracting data..."
tar xzf TCGA-TILs.tar.gz --strip-components=1 --exclude="*/README.md"

echo "[champkit] Removing .tar.gz file (it is no longer needed)"
rm TCGA-TILs.tar.gz

echo "[champkit] Generating classmap file..."
printf "til-negative\ntil-positive\n" > images-tcga-tils/pancancer/classmap.txt

echo "[champkit] Done!"
