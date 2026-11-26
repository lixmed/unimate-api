from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from api import ARTIFACT_PATH, FlatCompatibilityRequest, FlatInput, StudentProfile, get_engine
from train_flat_model import LABEL_MAP, LABEL_NAMES, UserFlatDataGenerator

matplotlib.use("Agg")

REPORT_DIR = Path("reports/user_flat_compatibility")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_SEED = 999


def get_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(EVAL_SEED)
    indices = np.arange(len(y_true))
    scores = []
    for _ in range(n_bootstrap):
        bootstrap_idx = rng.choice(indices, size=len(indices), replace=True)
        scores.append(accuracy_score(y_true[bootstrap_idx], y_pred[bootstrap_idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def build_holdout_dataset() -> pd.DataFrame:
    generator = UserFlatDataGenerator(ARTIFACT_PATH)
    generator.rng = np.random.default_rng(EVAL_SEED)
    return generator.generate_samples()


def build_sample_payload() -> FlatCompatibilityRequest:
    data = joblib.load(ARTIFACT_PATH)
    source_pool = data["males_pool"]
    current_user_row = source_pool.iloc[0]

    current_user = StudentProfile(
        student_id="demo_user_001",
        gender=str(current_user_row["Gender"]),
        age=int(current_user_row["Age"]),
        budget_range=float(current_user_row["Budget range"]),
        sleep_code=int(current_user_row["sleep_code"]),
        faculty=str(current_user_row["Faculty / Study Field"]),
        hometown_governorate=str(current_user_row["Hometown Governorate"]),
        wants_roommate=int(current_user_row.get("wants_roommate", 1)),
    )

    flats = []
    for offset, flat_size in enumerate([1, 3], start=1):
        members = []
        for idx in range(offset, offset + flat_size):
            row = source_pool.iloc[idx]
            members.append(
                StudentProfile(
                    student_id=f"flat_member_{idx}",
                    gender=str(row["Gender"]),
                    age=int(row["Age"]),
                    budget_range=float(row["Budget range"]),
                    sleep_code=int(row["sleep_code"]),
                    faculty=str(row["Faculty / Study Field"]),
                    hometown_governorate=str(row["Hometown Governorate"]),
                    wants_roommate=int(row.get("wants_roommate", 1)),
                )
            )
        flats.append(FlatInput(flat_id=f"demo_flat_{flat_size}", students=members))

    return FlatCompatibilityRequest(current_user=current_user, flats=flats, sort_descending=True)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def write_report(
    predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    report_dict: dict,
    cm_df: pd.DataFrame,
    metadata: dict,
    ci_low: float,
    ci_high: float,
    weighted_auc: float,
    sample_payload: FlatCompatibilityRequest,
    sample_response: dict,
) -> None:
    top_features = feature_importance_df[["feature", "importance"]].head(5).to_dict(orient="records")
    sample_input_json = json.dumps(sample_payload.model_dump(), indent=2)
    sample_output_json = json.dumps(sample_response, indent=2)

    with open(REPORT_DIR / "UNIMATE_COMPATIBILITY_REPORT.md", "w", encoding="utf-8") as file_obj:
        file_obj.write("# UniMate User-to-Flat Compatibility Model Report\n\n")
        file_obj.write("## 1. Executive Summary\n")
        file_obj.write(
            "This model scores how compatible the current user is with the residents already living in each flat.\n\n"
        )
        file_obj.write(f"- **Validation Accuracy:** {metadata['validation_accuracy']:.2%}\n")
        file_obj.write(f"- **Validation Weighted F1:** {metadata['validation_weighted_f1']:.4f}\n")
        file_obj.write(
            f"- **Holdout Accuracy (seed {EVAL_SEED}):** "
            f"{accuracy_score(predictions_df['true_label'], predictions_df['pred_label']):.2%}\n"
        )
        file_obj.write(f"- **95% Accuracy Confidence Interval:** [{ci_low:.2%}, {ci_high:.2%}]\n")
        file_obj.write(f"- **Weighted One-vs-Rest ROC-AUC:** {weighted_auc:.4f}\n\n")

        file_obj.write("## 2. Model Architecture\n")
        file_obj.write("```mermaid\n")
        file_obj.write("flowchart LR\n")
        file_obj.write("    A[Current user JSON] --> C[User-to-flat feature builder]\n")
        file_obj.write("    B[Flat residents JSON] --> C\n")
        file_obj.write("    C --> D[Distance and social-fit features]\n")
        file_obj.write("    D --> E[XGBoost multiclass classifier]\n")
        file_obj.write("    E --> F[weak / moderate / strong probabilities]\n")
        file_obj.write("    F --> G[overall score and ranking]\n")
        file_obj.write("```\n\n")
        file_obj.write(
            "The feature builder mixes direct user-to-resident signals such as budget distance, sleep compatibility, "
            "faculty similarity, hometown similarity, and encoded distance metrics with context features from the flat itself.\n\n"
        )

        file_obj.write("## 3. Loss Function and Training Dynamics\n")
        file_obj.write(
            "The classifier is trained with **multiclass log loss (`mlogloss`)**. "
            "That loss punishes confident wrong predictions more than uncertain ones, which is important when ranking candidate flats.\n\n"
        )
        file_obj.write("![Training and Validation Loss](loss_curve.png)\n\n")
        file_obj.write(
            "The chart shows how training loss and validation loss evolve across boosting rounds. "
            "A small gap suggests the model is learning separable compatibility patterns without a major overfitting jump.\n\n"
        )

        file_obj.write("## 4. Classification Metrics\n")
        file_obj.write(dataframe_to_markdown_table(pd.DataFrame(report_dict).transpose().reset_index()) + "\n\n")
        file_obj.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        file_obj.write(dataframe_to_markdown_table(cm_df.reset_index()) + "\n\n")

        file_obj.write("## 5. Charts and Insights\n")
        file_obj.write("### 5.1 Score Separation by Class\n")
        file_obj.write("![Score Distribution](score_distribution_by_class.png)\n\n")
        file_obj.write("### 5.2 Precision-Recall Curves\n")
        file_obj.write("![Precision Recall](pr_curves.png)\n\n")
        file_obj.write("### 5.3 ROC Curves\n")
        file_obj.write("![ROC Curves](roc_curves.png)\n\n")
        file_obj.write("### 5.4 Accuracy by Flat Size\n")
        file_obj.write("![Accuracy by Flat Size](accuracy_by_flat_size.png)\n\n")
        file_obj.write("### 5.5 Feature Importance\n")
        file_obj.write("![Feature Importance](feature_importance.png)\n\n")
        file_obj.write("Top drivers from the fitted model:\n\n")
        for item in top_features:
            file_obj.write(f"- **{item['feature']}**: {item['importance']:.4f}\n")
        file_obj.write("\n")
        file_obj.write(
            "Interpretation: strong predictions usually appear when the current user is close to the flat on the encoded distance features "
            "and does not meaningfully damage the flat's combined cohesion after joining.\n\n"
        )

        file_obj.write("## 6. API Contract\n")
        file_obj.write("### Sample Input\n")
        file_obj.write(f"```json\n{sample_input_json}\n```\n\n")
        file_obj.write("### Sample Output\n")
        file_obj.write(f"```json\n{sample_output_json}\n```\n\n")

def evaluate() -> None:
    engine = get_engine()
    with open(Path("artifacts/user_flat_compatibility_model/model_metadata.json"), "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    holdout_df = build_holdout_dataset()

    X_eval = holdout_df.drop(columns=["label", "pool"])
    y_true = holdout_df["label"].map(LABEL_MAP).to_numpy()
    y_probs = engine.model.predict_proba(X_eval)
    y_pred = engine.model.predict(X_eval)

    results_df = holdout_df.copy()
    results_df["true_label"] = y_true
    results_df["pred_label"] = y_pred
    results_df["true_class"] = holdout_df["label"]
    results_df["pred_class"] = [LABEL_NAMES[idx] for idx in y_pred]
    results_df["overall_score"] = 0.5 * y_probs[:, 1] + y_probs[:, 2]
    results_df["p_weak"] = y_probs[:, 0]
    results_df["p_moderate"] = y_probs[:, 1]
    results_df["p_strong"] = y_probs[:, 2]
    results_df.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    report_dict = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(REPORT_DIR / "classification_metrics.csv")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.to_csv(REPORT_DIR / "classification_confusion_matrix.csv")

    ci_low, ci_high = get_bootstrap_ci(y_true, y_pred)
    weighted_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x="true_class", y="overall_score", order=LABEL_NAMES)
    plt.title("Overall Score Distribution by True Class")
    plt.savefig(REPORT_DIR / "score_distribution_by_class.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(LABEL_NAMES):
        precision, recall, _ = precision_recall_curve((y_true == idx).astype(int), y_probs[:, idx])
        plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(REPORT_DIR / "pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve((y_true == idx).astype(int), y_probs[:, idx])
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(REPORT_DIR / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    accuracy_by_size = (
        results_df.groupby("flat_size")
        .apply(lambda frame: accuracy_score(frame["true_label"], frame["pred_label"]))
        .reset_index(name="accuracy")
    )
    accuracy_by_size.to_csv(REPORT_DIR / "accuracy_by_flat_size.csv", index=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=accuracy_by_size, x="flat_size", y="accuracy", color="#4C78A8")
    plt.ylim(0, 1)
    plt.title("Accuracy by Flat Size")
    plt.savefig(REPORT_DIR / "accuracy_by_flat_size.png", dpi=200, bbox_inches="tight")
    plt.close()

    train_loss = metadata["loss_curve"]["train_logloss"]
    val_loss = metadata["loss_curve"]["validation_logloss"]
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="train_logloss")
    plt.plot(val_loss, label="validation_logloss")
    plt.xlabel("Boosting Round")
    plt.ylabel("Multiclass Log Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(REPORT_DIR / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    feature_importance_df = pd.DataFrame(
        {
            "feature": engine.feature_names,
            "importance": engine.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(15), x="importance", y="feature", color="#59A14F")
    plt.title("Top Feature Importances")
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    sample_payload = build_sample_payload()
    sample_response = engine.score_batch(sample_payload)

    write_report(
        predictions_df=results_df,
        feature_importance_df=feature_importance_df,
        report_dict=report_dict,
        cm_df=cm_df,
        metadata=metadata,
        ci_low=ci_low,
        ci_high=ci_high,
        weighted_auc=weighted_auc,
        sample_payload=sample_payload,
        sample_response=sample_response,
    )

    print(f"Evaluation complete. Report generated in {REPORT_DIR}")


if __name__ == "__main__":
    evaluate()
