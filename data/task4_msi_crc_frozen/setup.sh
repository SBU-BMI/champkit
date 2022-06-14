#!/usr/bin/env bash
#
# Download and prepare data from Kather et al., 2019.

set -eu

echo
echo
echo "[champkit] ----------------- SETUP TASK 4 -----------------"

# Get directory of current script.
here="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

DOWNLOAD_DIR="downloaded-zips"
IMAGE_DIR="images"
CLASSMAP_FILE="classmap.txt"
wget_args="--no-clobber --content-disposition --quiet --show-progress"
unzip_args="-q -n"

# Note we use () to enclose the function instead of {}. We do this to run the function
# inside its own subshell. See https://unix.stackexchange.com/a/612614 for more info.
do_task4() (
    mkdir -p $here/$DOWNLOAD_DIR
    cd $here/$DOWNLOAD_DIR
    # Download Task 4 data (CRC frozen)
    echo "[champkit] Downloading task 4 data..."
    wget $wget_args https://zenodo.org/record/2532612/files/CRC_KR_TEST_MSIMUT.zip?download=1
    wget $wget_args https://zenodo.org/record/2532612/files/CRC_KR_TEST_MSS.zip?download=1
    wget $wget_args https://zenodo.org/record/2532612/files/CRC_KR_TRAIN_MSIMUT.zip?download=1
    wget $wget_args https://zenodo.org/record/2532612/files/CRC_KR_TRAIN_MSS.zip?download=1

    echo "[champkit] Verifying integrity of downloaded files..."
    md5sum --check ../md5sums.txt

    echo "[champkit] Extracting files to $here/$IMAGE_DIR"
    cd ..
    mkdir -p $IMAGE_DIR
    mkdir -p $IMAGE_DIR/trainval $IMAGE_DIR/test

    unzip $unzip_args $DOWNLOAD_DIR/CRC_KR_TEST_MSIMUT.zip -d $IMAGE_DIR/test
    unzip $unzip_args $DOWNLOAD_DIR/CRC_KR_TEST_MSS.zip -d $IMAGE_DIR/test

    unzip $unzip_args $DOWNLOAD_DIR/CRC_KR_TRAIN_MSIMUT.zip -d $IMAGE_DIR/trainval
    unzip $unzip_args $DOWNLOAD_DIR/CRC_KR_TRAIN_MSS.zip -d $IMAGE_DIR/trainval

    # Make classmap file.
    printf "MSS\nMSIMUT\n" > $IMAGE_DIR/$CLASSMAP_FILE

    echo "[champkit] Removing .zip files (they are no longer needed)"
    rm -r $DOWNLOAD_DIR
)

# Download and setup task datasets if they are not done already. -----------
echo "[champkit] Seeing if I should download task 4"
if [ ! -d $here/$IMAGE_DIR ]; then
    do_task4
fi

if [ -d $here/$DOWNLOAD_DIR ]; then
    echo "[champkit] Removing $here/$DOWNLOAD_DIR"
    rm -r $here/$DOWNLOAD_DIR
fi

# Make train/val splits if we haven't done it already. -----------------
if [ ! -f $here/trainval-partitions.csv ]; then
    echo "[champkit] Splitting all training sets into train/val partitions..."
    (cd $here; python ../task3_msi_crc_ffpe/split-data.py $IMAGE_DIR/trainval)
fi

echo "[champkit] Done!"
