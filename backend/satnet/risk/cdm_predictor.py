"""CDM-based collision-risk prediction using the trained XGBoost model."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


_MODEL_DIR = Path(__file__).parent / "ml"


@dataclass(frozen=True)
class CDMPrediction:
    """Result of an ML-based collision-risk prediction for one event."""
    event_id: str | int
    log10_risk: float
    probability: float
    risk_level: str
    risk_reason: str


class CDMRiskPredictor:
    """Load and run the XGBoost model with training-compatible preprocessing."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self._model: xgb.XGBRegressor | None = None
        self._scaler = None
        self._label_encoders: dict = {}
        self._col_info: dict = {}
        self._feature_meta: dict = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        model_path = self.model_dir / "xgboost_best.json"
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        preprocess_path = self.model_dir / "preprocess_artifacts.pkl"
        if preprocess_path.exists():
            with open(preprocess_path, "rb") as artifact_file:
                artifacts = pickle.load(artifact_file)
            self._scaler = artifacts.get("scaler")
            self._label_encoders = artifacts.get("encoders", {})
            self._col_info = artifacts.get("col_info", {})

        feature_path = self.model_dir / "feature_meta.json"
        if feature_path.exists():
            with open(feature_path, "r") as metadata_file:
                self._feature_meta = json.load(metadata_file)
        self._loaded = True

    def _preprocess(self, cdm_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same fill, encoding, and scaling steps used in training."""
        df = cdm_df.copy()
        numeric_features = self._col_info.get("numeric_features", [])
        categorical_features = self._col_info.get("categorical_features", [])

        if "event_id" in df.columns:
            present_numeric = [c for c in numeric_features if c in df.columns]
            if present_numeric:
                df[present_numeric] = df.groupby("event_id")[present_numeric].transform(
                    lambda values: values.ffill().bfill()
                )

        for column in numeric_features:
            if column in df.columns and df[column].isnull().any():
                finite_values = df[column].dropna()
                median = finite_values.median() if not finite_values.empty else 0
                df[column] = df[column].fillna(median)

        for column in categorical_features:
            if column in df.columns and df[column].isnull().any():
                mode = df[column].mode()
                df[column] = df[column].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

        for column, encoder in self._label_encoders.items():
            if column not in df.columns:
                continue
            known = set(encoder.classes_)
            df[column] = df[column].astype(str).map(
                lambda value: encoder.transform([value])[0]
                if value in known else -1
            )

        if self._scaler is not None:
            present_numeric = [c for c in numeric_features if c in df.columns]
            if present_numeric:
                for column in present_numeric:
                    df[column] = df[column].replace([np.inf, -np.inf], np.nan)
                    finite_values = df[column].dropna()
                    median = finite_values.median() if not finite_values.empty else 0
                    df[column] = df[column].fillna(median)
                df[present_numeric] = self._scaler.transform(df[present_numeric])
                df[present_numeric] = df[present_numeric].replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0)
        return df

    def predict(self, cdm_df: pd.DataFrame) -> list[CDMPrediction]:
        """Predict collision risk for each event in a training-schema CDM frame."""
        self._ensure_loaded()
        feature_cols = self._feature_meta.get("feature_cols", [])
        aggregate_names = self._feature_meta.get("agg_columns", [])
        if not feature_cols:
            raise ValueError("Feature metadata is missing feature_cols")
        if "event_id" not in cdm_df.columns:
            raise ValueError("CDM CSV must contain an event_id column")
        missing = [column for column in feature_cols if column not in cdm_df.columns]
        if missing:
            raise ValueError(
                "CDM CSV is missing required model features: "
                + ", ".join(missing[:10])
            )

        df = self._preprocess(cdm_df)
        rows = []
        event_ids = []
        for event_id, group in df.groupby("event_id"):
            if "time_to_tca" in group.columns:
                group = group.sort_values("time_to_tca", ascending=False)
            row = {}
            for column in feature_cols:
                values = group[column].to_numpy(dtype=np.float64)
                finite_values = values[np.isfinite(values)]
                if len(finite_values) == 0:
                    for suffix in ("mean", "std", "min", "max", "last", "first", "trend"):
                        row[f"{column}_{suffix}"] = 0.0
                    continue
                row[f"{column}_mean"] = np.nanmean(values)
                row[f"{column}_std"] = np.nanstd(values) if len(values) > 1 else 0.0
                row[f"{column}_min"] = np.nanmin(values)
                row[f"{column}_max"] = np.nanmax(values)
                row[f"{column}_last"] = values[-1] if len(values) else 0.0
                row[f"{column}_first"] = values[0] if len(values) else 0.0
                if len(values) > 1:
                    positions = np.arange(len(values), dtype=np.float64)
                    centered = positions - positions.mean()
                    row[f"{column}_trend"] = np.nansum(
                        centered * (values - np.nanmean(values))
                    ) / (np.nansum(centered ** 2) + 1e-10)
                else:
                    row[f"{column}_trend"] = 0.0
            row["cdm_count"] = len(group)
            row["time_span"] = (
                group["time_to_tca"].max() - group["time_to_tca"].min()
                if "time_to_tca" in group.columns else 0.0
            )
            row["min_time_to_tca"] = (
                group["time_to_tca"].min() if "time_to_tca" in group.columns else 0.0
            )
            rows.append(row)
            event_ids.append(event_id)

        aggregate = pd.DataFrame(rows)
        for column in aggregate_names:
            if column not in aggregate.columns:
                aggregate[column] = 0.0
        features = np.nan_to_num(
            aggregate[aggregate_names].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        predictions = self._model.predict(features)
        results = []
        for prediction, event_id in zip(predictions, event_ids):
            log10_risk = float(prediction)
            probability = float(10 ** log10_risk)
            level, reason = self._classify(probability)
            results.append(CDMPrediction(event_id, log10_risk, probability, level, reason))
        return results

    @staticmethod
    def _classify(probability: float) -> tuple[str, str]:
        if probability > 1e-4:
            return "RED", f"ML risk={probability:.2e} >= 1e-4: maneuver strongly recommended"
        if probability > 1e-5:
            return "YELLOW", f"ML risk={probability:.2e} >= 1e-5: monitor closely"
        if probability > 1e-7:
            return "YELLOW", f"ML risk={probability:.2e} >= 1e-7: routine monitoring"
        return "GREEN", f"ML risk={probability:.2e}: negligible risk"

    def is_available(self) -> bool:
        """Return whether the model artifacts load successfully."""
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False
