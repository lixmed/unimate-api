from __future__ import annotations

import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from scipy import sparse
from sklearn.metrics import pairwise_distances

from model_utils import extract_flat_features

ARTIFACT_PATH = Path("artifacts/unimate_matcher.joblib")
FLAT_MODEL_DIR = Path("artifacts/flat_model")
MAX_BATCH_SIZE = 500

app = FastAPI(title="UniMate Flat Scoring API", version="2.0.0")


class StudentProfile(BaseModel):
    student_id: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    age: int = Field(..., ge=17, le=40)
    budget_range: float = Field(..., ge=0.0, le=1.0)
    sleep_code: int = Field(..., ge=0, le=2)
    faculty: str = Field(..., min_length=1)
    hometown_governorate: str = Field(..., min_length=1)
    wants_roommate: int = Field(1, ge=0, le=1)


class FlatInput(BaseModel):
    flat_id: str = Field(..., min_length=1)
    students: list[StudentProfile] = Field(..., min_length=1, max_length=20)


class FlatScoreBatchRequest(BaseModel):
    flats: list[FlatInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    include_pair_scores: bool = Field(False, description="Include pair-level details for each scored flat")
    sort_descending: bool = Field(True, description="Sort returned flat scores by overall score descending")


class FlatScoreEngine:
    def __init__(self, artifact_path: Path, model_dir: Path) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")

        # Base artifact for encoders and pools
        data: dict[str, Any] = joblib.load(artifact_path)
        self.male_ct = data["male_ct"]
        self.female_ct = data["female_ct"]
        self.male_w = np.asarray(data["male_feature_weights_vector"])
        self.female_w = np.asarray(data["female_feature_weights_vector"])
        self.feature_columns = data["feature_columns"]
        self.max_budget_diff = float(data["MAX_BUDGET_DIFF"])

        # Model artifacts
        self.xgb_model = joblib.load(model_dir / "xgb_model.joblib")
        with open(model_dir / "feature_names.json", "r") as f:
            self.xgb_feature_names = json.load(f)
        with open(model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)

    def _get_weighted_matrix(self, df: pd.DataFrame, gender: str) -> np.ndarray:
        if gender.lower() == "male":
            ct, w = self.male_ct, self.male_w
        else:
            ct, w = self.female_ct, self.female_w

        X = ct.transform(df[self.feature_columns])
        Xw = X.multiply(w) if sparse.issparse(X) else X * w
        return Xw.tocsr() if sparse.issparse(Xw) else Xw

    def score_flat(self, flat: FlatInput) -> dict[str, Any]:
        students = [s.model_dump() for s in flat.students]
        
        # Map pydantic fields to expected feature column names
        mapping = {
            "age": "Age",
            "budget_range": "Budget range",
            "sleep_code": "sleep_code",
            "faculty": "Faculty / Study Field",
            "hometown_governorate": "Hometown Governorate"
        }
        
        # Create DataFrame with correct column names for the transformer
        df_for_ct = pd.DataFrame([
            {mapping[k]: v for k, v in s.items() if k in mapping}
            for s in students
        ])
        
        # Base DataFrame for other logic
        df = pd.DataFrame(students)

        # Check gender consistency
        genders = df["gender"].str.strip().str.capitalize().unique()
        if len(genders) > 1:
            return {"error": "Mixed gender groups are not supported for AI scoring"}
        gender = genders[0]

        # 1. Base Feature Extraction
        features = extract_flat_features(students, self.max_budget_diff)

        # 2. Distance Features
        X_weighted = self._get_weighted_matrix(df_for_ct, gender)
        dists = pairwise_distances(X_weighted, metric="euclidean")
        upper_triangle = dists[np.triu_indices_from(dists, k=1)]
        
        features["avg_dist"] = float(np.mean(upper_triangle))
        features["max_dist"] = float(np.max(upper_triangle))
        features["min_dist"] = float(np.min(upper_triangle))
        features["weakest_pair_score"] = 1 / (1 + features["max_dist"])
        features["penalty_factor"] = features["weakest_pair_score"] ** 2

        # 3. Model Scoring
        # Ensure features are in the correct order for XGBoost
        X_input = pd.DataFrame([features])[self.xgb_feature_names]
        probs = self.xgb_model.predict_proba(X_input)[0] # [weak, moderate, strong]
        
        # We define the "overall score" as a weighted sum of probabilities
        # to provide a continuous scale from 0 to 1
        overall_score = float(0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2])
        
        # Classification
        # We use the model's direct prediction for the label to maximize accuracy,
        # but keep the continuous overall_score for UI/sorting.
        y_pred = int(self.xgb_model.predict(X_input)[0])
        label_map_inv = {0: "weak", 1: "moderate", 2: "strong"}
        label = label_map_inv[y_pred]

        return {
            "flat_id": flat.flat_id,
            "overall_score": round(overall_score, 4),
            "compatibility_label": label,
            "probabilities": {
                "weak": round(float(probs[0]), 4),
                "moderate": round(float(probs[1]), 4),
                "strong": round(float(probs[2]), 4)
            },
            "metrics": {
                "avg_budget_diff": round(features["avg_budget_diff"], 4),
                "avg_sleep_compatibility": round(features["avg_sleep_compat"], 4),
                "avg_hometown_similarity": round(features["avg_hometown_sim"], 4),
                "avg_faculty_similarity": round(features["avg_faculty_sim"], 4),
                "weakest_pair_score": round(features["weakest_pair_score"], 4),
                "all_pairs_within_budget_threshold": features["budget_pass_rate"] == 1.0
            }
        }

    def score_batch(self, payload: FlatScoreBatchRequest) -> dict[str, Any]:
        scored_flats = []
        rejected_flats = []

        for flat in payload.flats:
            result = self.score_flat(flat)
            if "error" in result:
                rejected_flats.append({
                    "flat_id": flat.flat_id,
                    "status": "rejected",
                    "reason": result["error"]
                })
            else:
                scored_flats.append(result)

        if payload.sort_descending:
            scored_flats.sort(key=lambda item: item["overall_score"], reverse=True)

        return {
            "request_count": len(payload.flats),
            "scored_count": len(scored_flats),
            "rejected_count": len(rejected_flats),
            "flat_scores": scored_flats,
            "rejected_flats": rejected_flats,
        }


@lru_cache(maxsize=1)
def get_engine() -> FlatScoreEngine:
    return FlatScoreEngine(ARTIFACT_PATH, FLAT_MODEL_DIR)


@app.post("/flat-scores")
def score_all_flats(payload: FlatScoreBatchRequest):
    return get_engine().score_batch(payload)
