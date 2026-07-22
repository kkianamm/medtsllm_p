#!/usr/bin/env python3
"""Validate PULSE JSONL coverage, parse status, and obvious label leakage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LEAKAGE_PATTERNS = {
    "NORM": re.compile(r"\bNORM\b|\bnormal ECG class\b", re.IGNORECASE),
    "MI": re.compile(r"\bMI class\b|\bmyocardial infarction class\b", re.IGNORECASE),
    "STTC": re.compile(r"\bSTTC\b|\bST/T change class\b", re.IGNORECASE),
    "CD": re.compile(r"\bCD class\b|\bconduction disturbance class\b", re.IGNORECASE),
    "HYP": re.compile(r"\bHYP\b|\bhypertrophy class\b", re.IGNORECASE),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    descriptions = read_jsonl(args.descriptions)

    manifest_ids = [int(record["ecg_id"]) for record in manifest]
    output_ids = [int(record["ecg_id"]) for record in descriptions]
    duplicate_manifest = sorted(key for key, count in Counter(manifest_ids).items() if count > 1)
    duplicate_output = sorted(key for key, count in Counter(output_ids).items() if count > 1)

    latest_by_id = {int(record["ecg_id"]): record for record in descriptions}
    missing = sorted(set(manifest_ids) - set(latest_by_id))
    extra = sorted(set(latest_by_id) - set(manifest_ids))
    errors = sorted(
        ecg_id for ecg_id, record in latest_by_id.items() if record.get("status") != "ok"
    )
    empty = sorted(
        ecg_id
        for ecg_id, record in latest_by_id.items()
        if record.get("status") == "ok" and not str(record.get("description", "")).strip()
    )

    leakage: dict[str, list[int]] = {key: [] for key in LEAKAGE_PATTERNS}
    for ecg_id, record in latest_by_id.items():
        text = str(record.get("description", ""))
        for label, pattern in LEAKAGE_PATTERNS.items():
            if pattern.search(text):
                leakage[label].append(ecg_id)

    summary = {
        "manifest_records": len(manifest),
        "description_lines": len(descriptions),
        "unique_description_ids": len(latest_by_id),
        "ok_records": sum(record.get("status") == "ok" for record in latest_by_id.values()),
        "missing_count": len(missing),
        "error_count": len(errors),
        "empty_count": len(empty),
        "extra_count": len(extra),
        "duplicate_manifest_count": len(duplicate_manifest),
        "duplicate_output_count": len(duplicate_output),
        "leakage_counts": {key: len(value) for key, value in leakage.items()},
        "examples": {
            "missing": missing[:20],
            "errors": errors[:20],
            "empty": empty[:20],
            "extra": extra[:20],
            "duplicate_output": duplicate_output[:20],
            "leakage": {key: value[:20] for key, value in leakage.items()},
        },
    }
    print(json.dumps(summary, indent=2))

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = any(
        [
            missing,
            errors,
            empty,
            duplicate_manifest,
        ]
    )
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
