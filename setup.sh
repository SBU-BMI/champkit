#!/usr/bin/env bash
#
# Setup ChampKit.
# Download and prepare benchmark datasets.

set -eu

# Get directory of current script.
here="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cwd=$(pwd)

if [ "$(realpath $cwd)" != $(realpath "$here") ]; then
    echo "[champkit] ERROR: the setup.sh script must be run from $here"
    echo "[champkit] Please go to that directory an run this script again."
    exit 1
fi

program_exists() {
  hash "$1" 2>/dev/null;
}

if ! program_exists "conda"; then
    echo "[champkit] ERROR: I cannot find conda."
    echo "[champkit]        If conda is not installed, please install it from https://github.com/conda-forge/miniforge"
    exit 2
fi

# Check if MHIST data is ready to go...
# Check if images.zip exists and prompt user if it does not.
if [[ ! -f data/task6_precancer_vs_benign_polyps/images.zip ]] || [[ ! -f data/task6_precancer_vs_benign_polyps/annotations.csv ]]; then
    echo "[champkit] ERROR: cannot find images.zip or annotations.csv for Task 6 (MHIST dataset)."
    echo "           Please download them from https://bmirds.github.io/MHIST/#accessing-dataset"
    echo "           and move them to"
    echo "           $here/data/task6_precancer_vs_benign_polyps/"
    echo
    echo "           Once you have done this, please rerun this script."
    exit 3
fi

# https://stackoverflow.com/a/70598193/5666087
conda_env_exists(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if ! conda_env_exists "champkit"; then
    echo "[champkit] Creating champkit conda environment..."
    conda env create -f environment.yml -n champkit
fi

echo "[champkit] Activating champkit conda environment"
eval "$(conda shell.bash hook)"
set +u
conda activate champkit
set -u

echo "[champkit] IMPORTANT: If this your first time running this, I will download ~75 GB of benchmark data now."
echo "[champkit] This will likely take over two hours."
echo "[champkit] Here we go!"

bash data/task1_tumor_notumor/setup.sh
bash data/task2_tils/setup.sh
bash data/task3_msi_crc_ffpe/setup.sh
bash data/task4_msi_crc_frozen/setup.sh
bash data/task5_msi_stad_ffpe/setup.sh
bash data/task6_precancer_vs_benign_polyps/setup.sh

echo
echo "[champkit] Finished preparing the benchmark datasets."
echo "[champkit] Now you can train and evaluate models on these benchmarks."
echo "[champkit] Enjoy!"
