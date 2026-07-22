"""PTB-XL whole-record classification dataset for MedTsLLM.

Records are mapped to the five diagnostic superclasses NORM, MI, STTC, CD,
and HYP. Optional PULSE-generated ECG-image descriptions can be loaded from a
JSON/JSONL file and appended to the existing patient-information prompt.
"""

from __future__ import annotations

import ast
import json
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX = {name: idx for idx, name in enumerate(SUPERCLASS_ORDER)}


class ClassificationDataset(BaseDataset, ABC):
    """Whole-record classification: one full window and one scalar label."""

    supported_tasks = ["classification"]

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "classification"

    def load_data(self):
        data = self.get_data()
        features = np.asarray(data["data"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        features = self.normalize_records(features)

        self.records = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.record_descriptions = data.get("descriptions")

    def normalize_records(self, features: np.ndarray) -> np.ndarray:
        if not self.config.data.normalize:
            return features

        n_records, n_steps, n_features = features.shape
        if self.normalizer is None:
            if self.split == "train":
                train_features = features
            else:
                train_features = np.asarray(self.get_data("train")["data"], dtype=np.float32)
            self.normalizer = StandardScaler().fit(
                train_features.reshape(-1, train_features.shape[-1])
            )
        return self.normalizer.transform(features.reshape(-1, n_features)).reshape(
            n_records, n_steps, n_features
        ).astype(np.float32)

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, idx):
        item = {"x_enc": self.records[idx], "labels": self.labels[idx]}
        if self.record_descriptions is not None:
            item["descriptions"] = self.record_descriptions[idx]
        return item

    def inverse_index(self, idx):
        return idx

    @property
    def n_points(self):
        return self.records.shape[0]

    @property
    def n_features(self):
        return self.records.shape[2]


class PTBXLClassificationDataset(ClassificationDataset):
    description = (
        "PTB-XL is a large dataset of 12-lead ECGs, each a 10-second recording. "
        "Recordings are labeled with one of five diagnostic superclasses: "
        "Normal ECG (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), "
        "Conduction Disturbance (CD), and Hypertrophy (HYP)."
    )
    task_description = (
        "Classify the following 12-lead ECG recording into one of five diagnostic "
        "categories: Normal, Myocardial Infarction, ST/T Change, Conduction "
        "Disturbance, or Hypertrophy."
    )

    sampling_rate = 100

    @property
    def n_classes(self):
        return len(SUPERCLASS_ORDER)

    def _fold_split(self, split: str) -> list[int]:
        if split == "train":
            return list(range(1, 9))
        if split == "val":
            return [9]
        return [10]

    @property
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def _data_root(self) -> Path:
        configured = self.dataset_config.get("root", "data/ptbxl")
        path = Path(str(configured)).expanduser()
        return path if path.is_absolute() else (self._repo_root / path).resolve()

    def _pulse_config(self) -> Any:
        return self.dataset_config.get("pulse", {})

    def _resolve_description_path(self, configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        return path if path.is_absolute() else (self._repo_root / path).resolve()

    def _load_pulse_descriptions(self) -> dict[int, str]:
        """Load compact descriptions keyed by ECG ID.

        Supported inputs:
        * JSONL records containing ``ecg_id`` and ``description``.
        * A JSON dictionary mapping ECG ID to a string or record.
        * A JSON list using the same record structure as JSONL.
        """
        pulse_cfg = self._pulse_config()
        if not pulse_cfg or not pulse_cfg.get("enabled", False):
            return {}

        configured_path = str(pulse_cfg.get("descriptions_path", "")).strip()
        if not configured_path:
            raise ValueError(
                "datasets.PTB-XL.pulse.enabled=true requires descriptions_path"
            )
        path = self._resolve_description_path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"PULSE description file not found: {path}")

        cache_key = str(path)
        if getattr(self, "_pulse_cache_key", None) == cache_key:
            return self._pulse_cache

        if path.suffix.lower() == ".jsonl":
            records: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid PULSE JSONL at {path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Expected a JSON object at {path}:{line_number}"
                        )
                    records.append(value)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = []
                for ecg_id, value in payload.items():
                    if isinstance(value, str):
                        records.append({"ecg_id": ecg_id, "description": value})
                    elif isinstance(value, dict):
                        records.append({"ecg_id": ecg_id, **value})
                    else:
                        raise ValueError(
                            f"Unsupported description value for ECG {ecg_id}: {type(value)}"
                        )
            else:
                raise ValueError(f"Unsupported PULSE description payload in {path}")

        descriptions: dict[int, str] = {}
        failed_ids: list[int] = []
        for record in records:
            if "ecg_id" not in record:
                continue
            ecg_id = int(record["ecg_id"])
            if record.get("status", "ok") != "ok":
                failed_ids.append(ecg_id)
                continue
            description = str(record.get("description", "")).strip()
            if description:
                # Last occurrence wins. This makes resumed JSONL jobs safe even if a
                # record was regenerated later.
                descriptions[ecg_id] = description

        self._pulse_cache_key = cache_key
        self._pulse_cache = descriptions
        self._pulse_failed_ids = failed_ids
        return descriptions

    @staticmethod
    def _patient_description(row: pd.Series) -> str:
        age: int | str = "unknown" if pd.isna(row.get("age")) else int(row["age"])
        sex_value = row.get("sex")
        sex = (
            "unknown"
            if pd.isna(sex_value)
            else ("male" if int(sex_value) == 0 else "female")
        )
        payload = json.dumps({"age": age, "sex": sex}, ensure_ascii=False)
        return f"Patient information: {payload}."

    def _combined_description(
        self,
        ecg_id: int,
        row: pd.Series,
        pulse_descriptions: dict[int, str],
    ) -> str:
        pulse_cfg = self._pulse_config()
        pulse_enabled = bool(pulse_cfg and pulse_cfg.get("enabled", False))
        include_demographics = bool(pulse_cfg.get("include_demographics", True))

        parts: list[str] = []
        if not pulse_enabled or include_demographics:
            parts.append(self._patient_description(row))

        if pulse_enabled:
            pulse_text = pulse_descriptions.get(ecg_id)
            if pulse_text is None:
                if pulse_cfg.get("strict", True):
                    raise KeyError(
                        f"Missing successful PULSE description for ecg_id={ecg_id}. "
                        "Run tools/pulse/validate_pulse_descriptions.py first or set strict=false."
                    )
                pulse_text = str(
                    pulse_cfg.get(
                        "missing_text",
                        "PULSE visual ECG findings are unavailable for this record.",
                    )
                )

            prefix = str(pulse_cfg.get("prefix", "")).strip()
            if prefix and not pulse_text.startswith(prefix):
                pulse_text = f"{prefix} {pulse_text}"
            max_chars = int(pulse_cfg.get("max_chars", 1600))
            pulse_text = pulse_text[:max_chars].strip()
            parts.append(pulse_text)

        return " ".join(part for part in parts if part).strip()

    def get_data(self, split=None):
        import wfdb

        split = split or self.split
        basepath = self._data_root
        db = pd.read_csv(basepath / "ptbxl_database.csv", index_col="ecg_id")
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

        statements = pd.read_csv(basepath / "scp_statements.csv", index_col=0)
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
        db["label"] = db["superclasses"].apply(
            lambda values: SUPERCLASS_TO_IDX[next(iter(values))]
        )
        db = db[db["strat_fold"].isin(self._fold_split(split))]

        filename_column = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        crop = int(self.history_len)
        pulse_descriptions = self._load_pulse_descriptions()

        signals: list[np.ndarray] = []
        labels: list[int] = []
        descriptions: list[str] = []
        for ecg_id_raw, row in db.iterrows():
            ecg_id = int(ecg_id_raw)
            signal, _ = wfdb.rdsamp(str(basepath / row[filename_column]))
            signal = np.asarray(signal, dtype=np.float32)

            n_steps = signal.shape[0]
            if n_steps >= crop:
                start = (n_steps - crop) // 2
                signal = signal[start : start + crop]
            else:
                padding = np.zeros(
                    (crop - n_steps, signal.shape[1]), dtype=np.float32
                )
                signal = np.concatenate([signal, padding], axis=0)

            signals.append(signal)
            labels.append(int(row["label"]))
            descriptions.append(
                self._combined_description(ecg_id, row, pulse_descriptions)
            )

        return {
            "data": np.stack(signals, axis=0),
            "labels": np.asarray(labels, dtype=np.int64),
            "descriptions": descriptions,
        }


ptbxl_datasets = {"classification": PTBXLClassificationDataset}
