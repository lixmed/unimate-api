from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, log_loss, pairwise_distances
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from api import ARTIFACT_PATH
from model_utils import (
    canonicalize_student,
    extract_user_flat_features,
    students_to_transformer_frame,
    summarize_distance_values,
)

RANDOM_SEED = 42
FLAT_SIZES = [1, 2, 3, 4, 5, 6]
SAMPLES_PER_CLASS_PER_SIZE = 40
MODEL_DIR = Path("artifacts/user_flat_compatibility_model")
LABEL_MAP = {"weak": 0, "moderate": 1, "strong": 2}
LABEL_NAMES = ["weak", "moderate", "strong"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)


class UserFlatDataGenerator:
    def __init__(self, artifact_path: Path):
        self.data = joblib.load(artifact_path)
        self.feature_columns = self.data["feature_columns"]
        self.max_budget_diff = float(self.data["MAX_BUDGET_DIFF"])
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.pools = [
            ("male", self.data["males_pool"], self.data["male_ct"], np.asarray(self.data["male_feature_weights_vector"])),
            ("female", self.data["females_pool"], self.data["female_ct"], np.asarray(self.data["female_feature_weights_vector"])),
        ]

    def _get_weighted_matrix(self, df: pd.DataFrame, ct, weights):
        transformed = ct.transform(df[self.feature_columns])
        weighted = transformed.multiply(weights) if sparse.issparse(transformed) else transformed * weights
        return weighted.tocsr() if sparse.issparse(weighted) else weighted

    def _student_from_row(self, row: pd.Series, student_id: str) -> dict[str, object]:
        return canonicalize_student(
            {
                "student_id": student_id,
                "gender": row["Gender"],
                "age": row["Age"],
                "budget_range": row["Budget range"],
                "sleep_code": row["sleep_code"],
                "faculty": row["Faculty / Study Field"],
                "hometown_governorate": row["Hometown Governorate"],
                "wants_roommate": row.get("wants_roommate", 1),
            }
        )

    def _sample_cohesive_flat(self, distance_matrix: np.ndarray, flat_size: int) -> list[int]:
        anchor_idx = int(self.rng.integers(0, distance_matrix.shape[0]))
        ranked = np.argsort(distance_matrix[anchor_idx])
        return ranked[:flat_size].tolist()

    def _sample_random_flat(self, pool_size: int, flat_size: int) -> list[int]:
        return self.rng.choice(pool_size, size=flat_size, replace=False).tolist()

    def _choose_candidate(self, flat_indices: list[int], distance_matrix: np.ndarray, strategy: str) -> int:
        pool_size = distance_matrix.shape[0]
        candidates = [idx for idx in range(pool_size) if idx not in flat_indices]
        candidate_distances = np.array([distance_matrix[idx, flat_indices].mean() for idx in candidates])
        order = np.argsort(candidate_distances)

        if strategy == "strong":
            selection = order[: max(5, len(order) // 8)]
        elif strategy == "weak":
            selection = order[-max(5, len(order) // 8) :]
        else:
            lower = int(0.35 * len(order))
            upper = max(lower + 1, int(0.65 * len(order)))
            selection = order[lower:upper]

        chosen_rank = int(self.rng.choice(selection))
        return candidates[chosen_rank]

    def _build_feature_row(
        self,
        pool_df: pd.DataFrame,
        weighted_matrix,
        flat_indices: list[int],
        user_idx: int,
        label: str,
        pool_name: str,
    ) -> dict[str, float]:
        flat_students = [
            self._student_from_row(pool_df.iloc[flat_idx], f"{pool_name}_flat_{flat_idx}")
            for flat_idx in flat_indices
        ]
        current_user = self._student_from_row(pool_df.iloc[user_idx], f"{pool_name}_user_{user_idx}")
        raw_features = extract_user_flat_features(current_user, flat_students, self.max_budget_diff)

        matrix_indices = [user_idx, *flat_indices]
        combined_matrix = weighted_matrix[matrix_indices]
        user_distances = pairwise_distances(combined_matrix[0:1], combined_matrix[1:], metric="euclidean")[0]
        flat_internal_matrix = pairwise_distances(combined_matrix[1:], metric="euclidean")
        combined_distances = pairwise_distances(combined_matrix, metric="euclidean")

        user_distance_stats = summarize_distance_values(user_distances)
        flat_distance_stats = summarize_distance_values(flat_internal_matrix[np.triu_indices_from(flat_internal_matrix, k=1)])
        combined_distance_stats = summarize_distance_values(combined_distances[np.triu_indices_from(combined_distances, k=1)])

        raw_features.update(
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
                "label": label,
                "pool": pool_name,
            }
        )
        return raw_features

    def generate_samples(self) -> pd.DataFrame:
        records: list[dict[str, float]] = []

        for pool_name, pool_df, ct, weights in self.pools:
            weighted_matrix = self._get_weighted_matrix(pool_df, ct, weights)
            distance_matrix = pairwise_distances(weighted_matrix, metric="euclidean")
            pool_size = len(pool_df)
            print(f"Generating {pool_name} samples from a pool of {pool_size} students...")

            for flat_size in FLAT_SIZES:
                print(f"  Building flat size {flat_size} samples...", flush=True)
                for _ in range(SAMPLES_PER_CLASS_PER_SIZE):
                    strong_flat = self._sample_cohesive_flat(distance_matrix, flat_size)
                    strong_user = self._choose_candidate(strong_flat, distance_matrix, "strong")
                    records.append(
                        self._build_feature_row(pool_df, weighted_matrix, strong_flat, strong_user, "strong", pool_name)
                    )

                    moderate_flat = self._sample_random_flat(pool_size, flat_size)
                    moderate_user = self._choose_candidate(moderate_flat, distance_matrix, "moderate")
                    records.append(
                        self._build_feature_row(
                            pool_df, weighted_matrix, moderate_flat, moderate_user, "moderate", pool_name
                        )
                    )

                    weak_flat = self._sample_cohesive_flat(distance_matrix, flat_size)
                    weak_user = self._choose_candidate(weak_flat, distance_matrix, "weak")
                    records.append(
                        self._build_feature_row(pool_df, weighted_matrix, weak_flat, weak_user, "weak", pool_name)
                    )

        return pd.DataFrame(records)


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=140,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )


def train() -> None:
    generator = UserFlatDataGenerator(ARTIFACT_PATH)
    dataset = generator.generate_samples()
    dataset.to_csv(MODEL_DIR / "training_dataset_snapshot.csv", index=False)

    X = dataset.drop(columns=["label", "pool"])
    y = dataset["label"].map(LABEL_MAP).astype(int)
    feature_names = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    history_model = build_model()
    history_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    history = history_model.evals_result()

    val_probs = history_model.predict_proba(X_val)
    val_pred = history_model.predict(X_val)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_records = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = build_model()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        fold_pred = fold_model.predict(X.iloc[test_idx])
        cv_records.append(
            {
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y.iloc[test_idx], fold_pred)),
                "weighted_f1": float(f1_score(y.iloc[test_idx], fold_pred, average="weighted")),
            }
        )

    final_model = build_model()
    final_model.fit(X, y, verbose=False)

    joblib.dump(final_model, MODEL_DIR / "compatibility_model.joblib")
    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as file_obj:
        json.dump(feature_names, file_obj, indent=2)

    metadata = {
        "model_name": "UniMate User-to-Flat Compatibility Model",
        "version": "3.0.0",
        "target": "current_user_to_flat",
        "label_names": LABEL_NAMES,
        "training_seed": RANDOM_SEED,
        "flat_sizes_seen": FLAT_SIZES,
        "samples_per_class_per_size": SAMPLES_PER_CLASS_PER_SIZE,
        "dataset_rows": int(len(dataset)),
        "dataset_columns": int(len(feature_names)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_weighted_f1": float(f1_score(y_val, val_pred, average="weighted")),
        "validation_log_loss": float(log_loss(y_val, val_probs)),
        "cross_validation": {
            "fold_metrics": cv_records,
            "mean_accuracy": float(np.mean([row["accuracy"] for row in cv_records])),
            "mean_weighted_f1": float(np.mean([row["weighted_f1"] for row in cv_records])),
        },
        "loss_curve": {
            "train_logloss": history["validation_0"]["mlogloss"],
            "validation_logloss": history["validation_1"]["mlogloss"],
        },
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print("Training complete.")
    print(json.dumps(metadata["cross_validation"], indent=2))


if __name__ == "__main__":
    train()
