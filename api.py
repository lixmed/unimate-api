from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from scipy import sparse
from sklearn.metrics import pairwise_distances

from model_utils import (
    canonicalize_student,
    extract_user_flat_features,
    normalize_gender,
    students_to_transformer_frame,
    summarize_distance_values,
)

ARTIFACT_PATH = Path("artifacts/unimate_matcher.joblib")
MODEL_DIR = Path("artifacts/user_flat_compatibility_model")
MAX_BATCH_SIZE = 500
DEFAULT_LABEL_NAMES = ["weak", "moderate", "strong"]

app = FastAPI(title="UniMate User-to-Flat Compatibility API", version="3.0.0")


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


class FlatCompatibilityRequest(BaseModel):
    current_user: StudentProfile
    flats: list[FlatInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    sort_descending: bool = Field(True, description="Sort returned flat scores by overall score descending")


class UserFlatCompatibilityEngine:
    def __init__(self, artifact_path: Path, model_dir: Path) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_dir}")

        data: dict[str, Any] = joblib.load(artifact_path)
        self.male_ct = data["male_ct"]
        self.female_ct = data["female_ct"]
        self.male_w = np.asarray(data["male_feature_weights_vector"])
        self.female_w = np.asarray(data["female_feature_weights_vector"])
        self.feature_columns = data["feature_columns"]
        self.max_budget_diff = float(data["MAX_BUDGET_DIFF"])

        self.model = joblib.load(model_dir / "compatibility_model.joblib")
        with open(model_dir / "feature_names.json", "r", encoding="utf-8") as file_obj:
            self.feature_names = json.load(file_obj)

        metadata_path = model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file_obj:
                self.model_metadata = json.load(file_obj)
        else:
            self.model_metadata = {"label_names": DEFAULT_LABEL_NAMES}

        self.label_names = self.model_metadata.get("label_names", DEFAULT_LABEL_NAMES)

    def _get_weighted_matrix(self, df: pd.DataFrame, gender: str):
        if gender.lower() == "male":
            ct, weights = self.male_ct, self.male_w
        else:
            ct, weights = self.female_ct, self.female_w

        transformed = ct.transform(df[self.feature_columns])
        weighted = transformed.multiply(weights) if sparse.issparse(transformed) else transformed * weights
        return weighted.tocsr() if sparse.issparse(weighted) else weighted

    def _validate_request(self, current_user: dict[str, Any], flat_students: list[dict[str, Any]]) -> str | None:
        if not flat_students:
            return "Flat must contain at least one student."

        flat_genders = {normalize_gender(student["gender"]) for student in flat_students}
        if len(flat_genders) > 1:
            return "Mixed gender flats are not supported for AI scoring."

        flat_gender = next(iter(flat_genders))
        if normalize_gender(current_user["gender"]) != flat_gender:
            return "Current user gender must match the flat residents' gender."

        current_user_id = str(current_user["student_id"]).strip()
        flat_student_ids = {str(student["student_id"]).strip() for student in flat_students}
        if current_user_id in flat_student_ids:
            return "Current user already exists inside this flat payload."

        return None

    def _build_feature_vector(self, current_user: dict[str, Any], flat_students: list[dict[str, Any]]) -> dict[str, float]:
        all_students = [current_user, *flat_students]
        df_for_ct = students_to_transformer_frame(all_students)
        weighted_matrix = self._get_weighted_matrix(df_for_ct, current_user["gender"])

        user_distances = pairwise_distances(weighted_matrix[0:1], weighted_matrix[1:], metric="euclidean")[0]
        flat_internal_matrix = pairwise_distances(weighted_matrix[1:], metric="euclidean")
        combined_matrix = pairwise_distances(weighted_matrix, metric="euclidean")

        user_flat_features = extract_user_flat_features(current_user, flat_students, self.max_budget_diff)
        user_distance_stats = summarize_distance_values(user_distances)
        flat_distance_stats = summarize_distance_values(flat_internal_matrix[np.triu_indices_from(flat_internal_matrix, k=1)])
        combined_distance_stats = summarize_distance_values(combined_matrix[np.triu_indices_from(combined_matrix, k=1)])

        user_flat_features.update(
            {
                "user_distance_mean": user_distance_stats["mean"],
                "user_distance_min": user_distance_stats["min"],
                "user_distance_max": user_distance_stats["max"],
                "flat_distance_mean": flat_distance_stats["mean"],
                "flat_distance_min": flat_distance_stats["min"],
                "flat_distance_max": flat_distance_stats["max"],
                "combined_distance_mean": combined_distance_stats["mean"],
                "combined_distance_min": combined_distance_stats["min"],
                "combined_distance_max": combined_distance_stats["max"],
                "combined_weakest_link_score": 1.0 / (1.0 + combined_distance_stats["max"]),
                "delta_distance_mean": combined_distance_stats["mean"] - flat_distance_stats["mean"],
            }
        )
        return user_flat_features

    def score_flat_for_user(self, current_user: StudentProfile, flat: FlatInput) -> dict[str, Any]:
        try:
            user_dict = canonicalize_student(current_user.model_dump())
            flat_students = [canonicalize_student(student.model_dump()) for student in flat.students]
            validation_error = self._validate_request(user_dict, flat_students)
            if validation_error:
                return {"error": validation_error}

            features = self._build_feature_vector(user_dict, flat_students)
            feature_frame = pd.DataFrame([features]).reindex(columns=self.feature_names, fill_value=0.0)
            probabilities = self.model.predict_proba(feature_frame)[0]
            predicted_index = int(self.model.predict(feature_frame)[0])
            predicted_label = self.label_names[predicted_index]
            overall_score = float(0.0 * probabilities[0] + 0.5 * probabilities[1] + 1.0 * probabilities[2])

            return {
                "flat_id": flat.flat_id,
                "status": "scored",
                "overall_score": round(overall_score, 4),
                "compatibility_label": predicted_label,
                "probabilities": {
                    label: round(float(probability), 4)
                    for label, probability in zip(self.label_names, probabilities)
                },
                "metrics": {
                    "flat_size": int(features["flat_size"]),
                    "user_budget_diff_mean": round(features["user_budget_diff_mean"], 4),
                    "user_sleep_compat_mean": round(features["user_sleep_compat_mean"], 4),
                    "user_hometown_sim_mean": round(features["user_hometown_sim_mean"], 4),
                    "user_faculty_sim_mean": round(features["user_faculty_sim_mean"], 4),
                    "user_distance_mean": round(features["user_distance_mean"], 4),
                    "user_distance_max": round(features["user_distance_max"], 4),
                    "flat_distance_mean": round(features["flat_distance_mean"], 4),
                    "combined_distance_mean": round(features["combined_distance_mean"], 4),
                    "combined_weakest_link_score": round(features["combined_weakest_link_score"], 4),
                    "user_budget_fit_pass_rate": round(features["user_budget_pass_rate"], 4),
                },
                "notes": [
                    "Score reflects how well the current user matches the residents already inside this flat.",
                    "Flat internal cohesion is used only as context, not as the final target being scored.",
                ],
            }
        except Exception as exc:
            return {"error": f"Scoring failed safely for this flat: {exc}"}

    def score_batch(self, payload: FlatCompatibilityRequest) -> dict[str, Any]:
        scored_flats: list[dict[str, Any]] = []
        rejected_flats: list[dict[str, Any]] = []

        for flat in payload.flats:
            result = self.score_flat_for_user(payload.current_user, flat)
            if "error" in result:
                rejected_flats.append(
                    {
                        "flat_id": flat.flat_id,
                        "status": "rejected",
                        "reason": result["error"],
                    }
                )
            else:
                scored_flats.append(result)

        if payload.sort_descending:
            scored_flats.sort(key=lambda item: item["overall_score"], reverse=True)

        return {
            "current_user_id": payload.current_user.student_id,
            "request_count": len(payload.flats),
            "scored_count": len(scored_flats),
            "rejected_count": len(rejected_flats),
            "flat_scores": scored_flats,
            "rejected_flats": rejected_flats,
        }


@lru_cache(maxsize=1)
def get_engine() -> UserFlatCompatibilityEngine:
    return UserFlatCompatibilityEngine(ARTIFACT_PATH, MODEL_DIR)


@app.post("/flat-scores")
def score_all_flats(payload: FlatCompatibilityRequest):
    return get_engine().score_batch(payload)
