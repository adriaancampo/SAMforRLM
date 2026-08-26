#!/usr/bin/env python
"""
End-to-end Python preparation pipeline for the publication analysis.

Usage
-----
python master.py --project-root "C:/path/to/project"

The selected project root must contain:
    Data/
    Checkpoints/
        sam_vit_b_01ec64.pth
        sam_vit_l_0b3195.pth
        sam_vit_h_4b8939.pth

Python writes all derived inputs for MATLAB below:
    Processed_Data/

By default, MATLAB is launched after Python preprocessing/inference finishes.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.preprocessing import preprocess_dataset
from src.sam_pipeline import sam_grain_pipeline, write_timing_csv


CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate all Python-derived inputs and run the MATLAB publication analysis."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Project directory containing Data/ and Checkpoints/.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Reuse existing Processed_Data/processed_* directories.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse existing Processed_Data/sam_outputs.",
    )
    parser.add_argument(
        "--skip-matlab",
        action="store_true",
        help="Do not launch MATLAB after Python finishes.",
    )
    parser.add_argument(
        "--matlab-command",
        default="matlab",
        help="MATLAB executable/command (default: matlab).",
    )
    return parser.parse_args()


def require_path(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def preprocess_all(project_root: Path, processed_root: Path):
    data_root = project_root / "Data"

    print("\n=== PREPROCESSING MULTICHANNEL DATA ===")
    preprocess_dataset(
        input_root=data_root,
        method="multichannel",
        gridsize=256,
        gridoverlap=10,
        outputfolder=processed_root / "processed_multichannel",
    )

    print("\n=== PREPROCESSING Cu RGB DATA ===")
    preprocess_dataset(
        input_root=data_root / "Cu_v1" / "Cu_v1",
        method="rgb",
        gridsize=640,
        gridoverlap=10,
        outputfolder=processed_root / "processed_rgb_2",
    )

    print("\n=== PREPROCESSING FeM RGB DATA ===")
    preprocess_dataset(
        input_root=data_root / "FeM_v1" / "FeM_v1",
        method="rgb",
        gridsize=640,
        gridoverlap=10,
        outputfolder=processed_root / "processed_rgb_3",
    )


def run_sam_all(project_root: Path, processed_root: Path):
    checkpoint_root = project_root / "Checkpoints"
    sam_output_root = processed_root / "sam_outputs"

    checkpoint_paths = {
        model: checkpoint_root / filename
        for model, filename in CHECKPOINTS.items()
    }
    for model, checkpoint in checkpoint_paths.items():
        require_path(checkpoint, f"{model} SAM checkpoint")

    rgb_inputs = {
        "Cu": processed_root / "processed_rgb_2",
        "FeM": processed_root / "processed_rgb_3",
    }
    multichannel_input = processed_root / "processed_multichannel"

    # Case study I: RGB input, no channel reduction.
    # Cu and FeM share an inference_<method>_<model> directory. Their
    # per-run timing records are merged into one explicit Dataset-tagged CSV.
    for model in ("vit_b", "vit_l", "vit_h"):
        combined_timing = []

        for dataset_name, input_dir in rgb_inputs.items():
            print(f"\n=== SAM {dataset_name} | none | {model} ===")
            records = sam_grain_pipeline(
                inputfolder=input_dir,
                outputfolder=sam_output_root,
                channel_method="none",
                sam_checkpoint=checkpoint_paths[model],
                sam_model_type=model,
                run_label=dataset_name,
            )
            combined_timing.extend(records)

        timing_path = (
            sam_output_root
            / f"inference_none_{model}"
            / "timing_per_image.csv"
        )
        write_timing_csv(timing_path, combined_timing)

    # Case study II: multichannel projection and PCA, all three encoders.
    for channel_method in ("proj", "pca"):
        for model in ("vit_b", "vit_l", "vit_h"):
            print(f"\n=== SAM multichannel | {channel_method} | {model} ===")
            sam_grain_pipeline(
                inputfolder=multichannel_input,
                outputfolder=sam_output_root,
                channel_method=channel_method,
                sam_checkpoint=checkpoint_paths[model],
                sam_model_type=model,
                run_label="Multichannel",
            )


def run_matlab(project_root: Path, matlab_command: str, code_root: Path):
    matlab_executable = shutil.which(matlab_command)
    if matlab_executable is None:
        raise RuntimeError(
            f"MATLAB executable '{matlab_command}' was not found on PATH. "
            "Run with --skip-matlab, or provide --matlab-command."
        )

    matlab_script = code_root / "matlab" / "MasterAnalysis.m"
    require_path(matlab_script, "MATLAB master script")

    env = os.environ.copy()
    env["PIRARD_PROJECT_ROOT"] = str(project_root)

    matlab_path = str(matlab_script).replace("'", "''")
    command = [
        matlab_executable,
        "-batch",
        f"run('{matlab_path}')",
    ]

    print("\n=== MATLAB PUBLICATION ANALYSIS ===")
    print(f"Project root: {project_root}")
    subprocess.run(command, env=env, check=True)


def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    code_root = Path(__file__).resolve().parent
    processed_root = project_root / "Processed_Data"

    require_path(project_root / "Data", "Data directory")
    require_path(project_root / "Checkpoints", "Checkpoints directory")

    processed_root.mkdir(parents=True, exist_ok=True)
    (project_root / "Figures_Publication").mkdir(parents=True, exist_ok=True)
    (project_root / "Tables_Publication").mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Publication pipeline")
    print("============================================================")
    print(f"Code root      : {code_root}")
    print(f"Project root   : {project_root}")
    print(f"Processed data : {processed_root}")

    if not args.skip_preprocessing:
        preprocess_all(project_root, processed_root)

    if not args.skip_inference:
        run_sam_all(project_root, processed_root)

    if not args.skip_matlab:
        run_matlab(project_root, args.matlab_command, code_root)

    print("\nPipeline complete.")
    print(f"Figures: {project_root / 'Figures_Publication'}")
    print(f"Tables : {project_root / 'Tables_Publication'}")


if __name__ == "__main__":
    main()
