"""Evaluate all model training runs in a directory.

This script creates a CSV file with the test-set performance of each model, prints a
summary of the best models, and saves a figure portraying the test-set performance of
all models.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml

try:
    from yaml import CSafeLoader as Loader
except Exception:
    from yaml import SafeLoader as Loader


class ChampKitException(Exception):
    ...


class ErrorOnValidation(ChampKitException):
    ...


def _prepare_df_for_run(directory: Path) -> pd.DataFrame:
    with open(directory / "args.yaml") as f:
        args = yaml.load(f, Loader=Loader)
    df = pd.DataFrame.from_dict(args, orient="index").T
    df.insert(0, "directory", directory)
    return df


def _get_results_json_from_stdout(stdout: str) -> Dict:
    """Process the stdout of 'validate.py' and return a python dict of the
    validation results.
    """
    stdout_lines = stdout.splitlines()
    try:
        start_idx = stdout_lines.index("--result")
    except ValueError:
        raise ChampKitException(
            "could not find --result in the output of validate.py... contact developer."
        )

    # example of this variable
    # ['{',
    #  '    "model": "resnet50",',
    #  '    "top1": 84.5001,',
    #  '    "top1_err": 15.4999,',
    #  '    "auroc": 0.9364033341407776,',
    #  '    "f1": 0.845001220703125,',
    #  '    "param_count": 23.51,',
    #  '    "img_size": 224,',
    #  '    "crop_pct": 0.95,',
    #  '    "interpolation": "bicubic"',
    #  '}']
    stdout_lines_maybe_results_json = stdout_lines[start_idx + 1 :]

    stop_idx = stdout_lines_maybe_results_json.index("}")
    json_str_with_results = " ".join(stdout_lines_maybe_results_json[: stop_idx + 1])

    # we have parsed the results into a python dict!
    # {'model': 'resnet50',
    #  'top1': 84.5001,
    #  'top1_err': 15.4999,
    #  'auroc': 0.9364033341407776,
    #  'f1': 0.845001220703125,
    #  'param_count': 23.51,
    #  'img_size': 224,
    #  'crop_pct': 0.95,
    #  'interpolation': 'bicubic'}
    results = json.loads(json_str_with_results)
    return results


def _run_one_evaluation(row: pd.Series) -> pd.DataFrame:
    """Run an evaluation given a row of the dataframe that contains info on all runs."""
    row = row.copy()  # make sure we don't modify original

    checkpoint = Path(row["directory"]) / "model_best.pth.tar"
    data_dir = Path(row["data_dir"])
    classmap_file = data_dir / "classmap.txt"
    if not classmap_file.exists():
        raise FileNotFoundError(f"cannot find the classmap file: {classmap_file}")

    print(f"[champkit]   checkpoint={checkpoint}")

    num_classes = int(row["num_classes"])
    print(f"[champkit]   num_classes={num_classes}")

    program_and_args = f"""
    {sys.executable} \
    validate.py \
    --model={row["model"]} \
    --checkpoint={checkpoint} \
    --batch-size=64 \
    --split=test \
    --num-classes={num_classes} \
    --class-map={classmap_file}""".strip()
    program_and_args += f' {row["data_dir"]}'

    p = subprocess.run(program_and_args.split(), capture_output=True, env=os.environ)

    if p.returncode != 0:
        print("** ERROR **" * 8)
        print(p.stderr.decode())
        print("Here is the command-line that errored:")
        print(" ".join(p.args))
        raise ErrorOnValidation(
            "Error on validation. Please see logs immediately above this."
        )

    tmp_results = _get_results_json_from_stdout(p.stdout.decode())

    results = dict(
        model=row["model"],
        data_dir=row["data_dir"],
        pretrained=row["pretrained"],
        checkpoint=checkpoint,
        classmap=classmap_file,
        num_classes=row["num_classes"],
        seed=row['seed'],
    )
    if results["num_classes"] > 100:
        raise ValueError("this script does not support num_classes>100")
    # Make a dataframe with one row.
    df = pd.DataFrame(results, index=[0])
    # Fill in the stats from evaluation.
    for stat in ["auroc", "f1", "fp", "fn", "tp", "tn", "fpr", "fnr", "tpr", "tnr", "accuracy"]:
        cols = [f"{stat}_cls{i:02d}" for i in range(num_classes)]
        df[cols] = tmp_results[stat]
    assert len(df) == 1
    return df


def run_all_evaluations(directory) -> pd.DataFrame:
    dirs = [p for p in Path(directory).glob("*") if p.is_dir()]
    dirs.sort()
    if not dirs:
        raise ChampKitException(f"no directories found in {directory}")
    print(f"[champkit] Found {len(dirs)} runs in {directory}")

    df = pd.concat((_prepare_df_for_run(d) for d in dirs), ignore_index=True)

    best_models = df.loc[:, "directory"] / "model_best.pth.tar"
    models_exist_mask = best_models.map(Path.exists)
    print(
        f"[champkit] Will evaluate the {models_exist_mask.sum()} runs with model_best.pth.tar."
    )
    df = df.loc[models_exist_mask, :].copy()
    if df.shape[0] == 0:
        raise ChampKitException("no model_best.pth.tar files found...")

    all_results: List[pd.DataFrame] = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"[champkit] evaluating run {i+1} of {df.shape[0]}...")
        print(f"[champkit]   run_dir={row['directory']}")
        print(f"[champkit]   model={row['model']}")
        print(f"[champkit]   pretrained={row['pretrained']}")
        print(f"[champkit]   data_dir={row['data_dir']}")
        state_dict = torch.load(
            Path(row["directory"]) / "model_best.pth.tar", map_location="cpu"
        )
        epoch = state_dict.get("epoch")
        if epoch is not None:
            print(f"[champkit]   epoch={epoch}")
        del state_dict
        result = _run_one_evaluation(row=row)
        print("[champkit]   Results:")
        for class_idx in range(result["num_classes"][0]):
            print(f"[champkit]     Class {class_idx}")
            print(f"[champkit]       Accuracy={result[f'accuracy_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       AUROC={result[f'auroc_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       F1={result[f'f1_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       FPR={result[f'fpr_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       FNR={result[f'fnr_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       TPR={result[f'tpr_cls{class_idx:02d}'][0]:0.3f}")
            print(f"[champkit]       TNR={result[f'tnr_cls{class_idx:02d}'][0]:0.3f}")
            print("[champkit]")

        result["epoch"] = epoch  # could be None but that's ok

        # Convert list or tuple to string to allow addition to dataframe.
        row = row.map(lambda p: str(p) if isinstance(p, (list, tuple)) else p)
        # Drop any names that are already in the dataframe.
        row = row[~row.index.isin(result.columns)]
        # Add these new values.
        row = row.to_frame(name=0).T  # Set index to 0...
        result = pd.concat((result, row), axis=1)
        assert len(result) == 1
        all_results.append(result)
        del result, row  # for our sanity
        print()

    df = pd.concat(all_results, axis=0, ignore_index=True)
    return df


def _print_summary(df: pd.DataFrame, output_pdf: str):
    print()
    print("***********************************")
    print("        SUMMARY OF EVALUATION      ")
    print("***********************************")
    print()

    num_classes = df["num_classes"][0]
    metric_info = {
        "auroc": {"columns": [f"auroc_cls{c:02d}" for c in range(num_classes)], "mode": "max"},
        "accuracy": {"columns": [f"accuracy_cls{c:02d}" for c in range(num_classes)], "mode": "max"},
        "tpr": {"columns": [f"tpr_cls{c:02d}" for c in range(num_classes)], "mode": "max"},
        "tnr": {"columns": [f"tnr_cls{c:02d}" for c in range(num_classes)], "mode":"max"},
        "fpr": {"columns": [f"fpr_cls{c:02d}" for c in range(num_classes)], "mode":"min"},
        "fnr": {"columns": [f"fnr_cls{c:02d}" for c in range(num_classes)], "mode":"min"},
    }

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 13))

    TOPK = 3
    best_ids = []
    for (metric, info), ax in zip(metric_info.items(), axes.flat):
        metric_values = df[info["columns"]].copy()
        mode = info["mode"]
        means_per_class = metric_values.mean(axis=1)
        metric_values[f"{metric}_mean"] = means_per_class

        print(f"***** {metric.upper()} *****")
        for col in metric_values.columns:
            if mode == "max":
                best_models = metric_values[col].nlargest(TOPK)
            elif mode == "min":
                best_models = metric_values[col].nsmallest(TOPK)
            else:
                raise NotImplementedError(f"unknown mode '{mode}'")

            print(f"Top {TOPK} models by {metric} -- '{col}'\t{best_models.index.tolist()}")
            best_ids.extend(best_models.index.tolist())
        print()

        metric_values["model_number"] = metric_values.index.copy()
        metric_values_melted = metric_values.melt(id_vars="model_number")
        sns.scatterplot(data=metric_values_melted, x="model_number", y="value", hue="variable", ax=ax)
        ax.set_title(f"{metric.title()} by model number")
        ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Model number (row in CSV from evaluation)")

    fig.tight_layout()
    print(f"[champkit] Saving plots of model performance to {output_pdf}")
    plt.savefig(output_pdf)

    best_ids = sorted(set(best_ids))
    print()
    print("*" * 40)
    print(f"Model checkpoints that were in the top {TOPK} of any evaluation metric:")
    print("  Please refer to the evaluation summary above for the best models per evaluation metric.")
    print("  NOTE: these are NOT sorted by performance. The models are sorted by their position in the evaluation CSV.")
    print()
    for model_number, checkpoint in df.loc[best_ids, "checkpoint"].iteritems():
        print(f"{str(model_number):>3s}\t{checkpoint}")
    print("*" * 40)
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--directory",
        required=True,
        help="Top-level directory that contains individual runs.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV with results. Default: champkit_evaluations_YYYYMMDDHHmmss.csv",
    )
    args = p.parse_args()

    if args.output_csv is None:
        args.output_csv = (
            f'champkit_evaluations_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        )

    print(f"[champkit] Evaluating runs in {args.directory}", flush=True)
    df = run_all_evaluations(args.directory)
    print(f"[champkit] Saving evaluation results to {args.output_csv}")
    df.to_csv(args.output_csv, index=False)

    output_pdf = Path(args.output_csv).with_suffix(".pdf")
    _print_summary(df=df, output_pdf=output_pdf)
    print("[champkit] Done!")
