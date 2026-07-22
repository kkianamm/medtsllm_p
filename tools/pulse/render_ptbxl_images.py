#!/usr/bin/env python3
"""Render deterministic 12-lead PTB-XL ECG images for PULSE.

The output is a standard 4x3 ECG layout plus a full-length lead-II rhythm strip.
Only signal and optional demographics are printed; diagnostic labels are never
written into the image.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb


SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX = {name: idx for idx, name in enumerate(SUPERCLASS_ORDER)}
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptbxl-root", type=Path, default=Path("data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ptbxl/pulse_images"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ptbxl/pulse_images/manifest.jsonl"),
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-demographics", action="store_true")
    return parser.parse_args()


def folds_for_split(split: str) -> list[int]:
    if split == "train":
        return list(range(1, 9))
    if split == "val":
        return [9]
    if split == "test":
        return [10]
    return list(range(1, 11))


def split_for_fold(fold: int) -> str:
    if fold <= 8:
        return "train"
    if fold == 9:
        return "val"
    return "test"


def load_index(root: Path, split: str) -> pd.DataFrame:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    statements = statements[statements["diagnostic"] == 1]

    def to_superclasses(codes: dict[str, float]) -> set[str]:
        classes: set[str] = set()
        for code in codes:
            if code in statements.index:
                value = statements.loc[code, "diagnostic_class"]
                if isinstance(value, str) and value in SUPERCLASS_TO_IDX:
                    classes.add(value)
        return classes

    db["superclasses"] = db["scp_codes"].apply(to_superclasses)
    db = db[db["superclasses"].apply(lambda values: len(values) == 1)].copy()
    db["label_name"] = db["superclasses"].apply(lambda values: next(iter(values)))
    db["label"] = db["label_name"].map(SUPERCLASS_TO_IDX)
    db = db[db["strat_fold"].isin(folds_for_split(split))]
    return db.sort_index()


def reorder_signal(signal: np.ndarray, signal_names: Iterable[str]) -> np.ndarray:
    normalized = {str(name).strip().upper(): i for i, name in enumerate(signal_names)}
    indices: list[int] = []
    for lead in LEAD_ORDER:
        key = lead.upper()
        if key not in normalized:
            raise ValueError(f"Required lead {lead!r} is missing; available={list(signal_names)}")
        indices.append(normalized[key])
    return signal[:, indices]


def configure_ecg_axis(ax: plt.Axes, duration: float, amplitude: float = 2.0) -> None:
    ax.set_xlim(0.0, duration)
    ax.set_ylim(-amplitude, amplitude)
    ax.set_facecolor("#fffafa")

    # 25 mm/s: 0.04 s small box and 0.20 s large box.
    ax.set_xticks(np.arange(0.0, duration + 1e-9, 0.20))
    ax.set_xticks(np.arange(0.0, duration + 1e-9, 0.04), minor=True)
    # 10 mm/mV: 0.1 mV small box and 0.5 mV large box.
    ax.set_yticks(np.arange(-amplitude, amplitude + 1e-9, 0.5))
    ax.set_yticks(np.arange(-amplitude, amplitude + 1e-9, 0.1), minor=True)
    ax.grid(which="major", color="#e8a0a0", linewidth=0.65, alpha=0.75)
    ax.grid(which="minor", color="#f4cccc", linewidth=0.30, alpha=0.70)
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_calibration(ax: plt.Axes) -> None:
    # 1 mV high, 0.20 s wide calibration pulse.
    x = np.array([0.02, 0.04, 0.04, 0.24, 0.24, 0.26])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) - 1.55
    ax.plot(x, y, color="black", linewidth=1.0)


def render_record(
    signal: np.ndarray,
    sampling_rate: int,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    duration = signal.shape[0] / sampling_rate
    if duration < 4.0:
        raise ValueError(f"ECG is too short for a 4x3 layout: duration={duration:.2f}s")

    segment_samples = signal.shape[0] // 4
    segment_duration = segment_samples / sampling_rate

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(4, 4, hspace=0.08, wspace=0.03, height_ratios=[1, 1, 1, 1.05])

    lead_to_idx = {lead: idx for idx, lead in enumerate(LEAD_ORDER)}
    for row_idx, row in enumerate(LAYOUT):
        for col_idx, lead in enumerate(row):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            configure_ecg_axis(ax, segment_duration)
            start = col_idx * segment_samples
            stop = start + segment_samples
            y = signal[start:stop, lead_to_idx[lead]]
            x = np.arange(y.size, dtype=np.float32) / sampling_rate
            ax.plot(x, y, color="black", linewidth=0.75)
            ax.text(
                0.015,
                0.90,
                lead,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": "#fffafa", "edgecolor": "none", "pad": 1.0},
            )
            if row_idx == 0 and col_idx == 0:
                draw_calibration(ax)

    rhythm_ax = fig.add_subplot(grid[3, :])
    configure_ecg_axis(rhythm_ax, duration)
    rhythm = signal[:, lead_to_idx["II"]]
    rhythm_time = np.arange(rhythm.size, dtype=np.float32) / sampling_rate
    rhythm_ax.plot(rhythm_time, rhythm, color="black", linewidth=0.75)
    rhythm_ax.text(
        0.004,
        0.90,
        "II rhythm strip",
        transform=rhythm_ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "#fffafa", "edgecolor": "none", "pad": 1.0},
    )

    fig.suptitle(title, fontsize=11, y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.ptbxl_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    if not (root / "ptbxl_database.csv").exists():
        raise FileNotFoundError(f"PTB-XL metadata not found under {root}")

    db = load_index(root, args.split)
    if args.limit is not None:
        db = db.iloc[: args.limit]

    filename_column = "filename_lr" if args.sampling_rate == 100 else "filename_hr"
    records: list[dict[str, object]] = []

    for position, (ecg_id, row) in enumerate(db.iterrows(), start=1):
        split = split_for_fold(int(row["strat_fold"]))
        image_path = output_dir / split / f"{int(ecg_id):05d}.png"

        source_path = root / str(row[filename_column])
        if args.overwrite or not image_path.exists():
            signal, fields = wfdb.rdsamp(str(source_path))
            signal = reorder_signal(np.asarray(signal, dtype=np.float32), fields["sig_name"])

            title = f"12-lead ECG | {args.sampling_rate} Hz | {signal.shape[0] / args.sampling_rate:.1f} s"
            if args.include_demographics:
                age = "unknown" if pd.isna(row.get("age")) else str(int(row["age"]))
                sex_value = row.get("sex")
                sex = "unknown" if pd.isna(sex_value) else ("male" if int(sex_value) == 0 else "female")
                title += f" | age {age} | {sex}"

            render_record(signal, args.sampling_rate, image_path, args.dpi, title)

        records.append(
            {
                "ecg_id": int(ecg_id),
                "split": split,
                "strat_fold": int(row["strat_fold"]),
                "image_path": str(image_path),
                "source_path": str(source_path),
                # Retained only for audit/evaluation. The PULSE generation script never
                # places these fields in the model prompt.
                "label": int(row["label"]),
                "label_name": str(row["label_name"]),
            }
        )
        if position % 100 == 0 or position == len(db):
            print(f"Rendered/verified {position}/{len(db)} records")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Manifest written to {manifest_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
