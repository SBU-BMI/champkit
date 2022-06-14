#!/usr/bin/env bash

set -eu

echo
echo
echo "[champkit] ----------------- SETUP TASK 6 -----------------"

# Get directory of current script.
here="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# Exit early if dataset seems to be ready.
if [ -d $here/images-split ]; then
    echo "[champkit] found $here/images-split directory... not doing anything because this dataset seems to be ready."
    exit 0
fi


# All paths in this script are relative to the script's directory.
cd $here

# Check if images.zip exists and prompt user if it does not.
if [[ ! -f images.zip ]] || [[ ! -f annotations.csv ]]; then
    echo "[champkit] ERROR: cannot find images.zip or annotations.csv for MHIST dataset."
    echo "           Please download them from https://bmirds.github.io/MHIST/#accessing-dataset"
    echo "           and move them to"
    echo "           $here"
    exit 1
fi

echo "[champkit] Verifying integrity of downloaded files..."
md5sum --check md5sums.txt

echo "[champkit] Unzipping data..."
unzip -q -n images.zip

# Let's not remove the zip file. It's small enough that it shouldn't matter, and the
# user downloaded it and moved it here... might seem offensive if we get rid of it
# without asking :)


echo "[champkit] Creating train/val/test partitions..."
python split-data.py

echo "[champkit] Generating classmap.txt"
printf "HP\nSSA\n" > images-split/classmap.txt

echo "[champkit] Done!"
