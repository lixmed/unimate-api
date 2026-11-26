# UniMate — User-to-Flat Compatibility API

UniMate is an AI-powered student housing platform. This repository is its **compatibility engine**: for a student looking for housing, it scores how well they would fit with the residents already living in each available flat, then ranks the flats.

Built with **FastAPI**, **XGBoost**, and **scikit-learn**. Deployed on Hugging Face Spaces: [lixmed/unimate-ai](https://huggingface.co/spaces/lixmed/unimate-ai).

## Results

| Metric | Value |
|---|---|
| Holdout accuracy | **97.29%** (1,440 holdout cases) |
| 95% confidence interval | 96.46% – 97.99% (bootstrap) |
| Weighted F1 | 0.973 |
| Weighted one-vs-rest ROC-AUC | 0.998 |
| Validation accuracy | 95.49% |

Per class: weak F1 0.980, moderate F1 0.962, strong F1 0.977. The full report with charts (confusion matrix, ROC and PR curves, loss curve, accuracy by flat size, feature importance) is in [`reports/user_flat_compatibility/UNIMATE_COMPATIBILITY_REPORT.md`](reports/user_flat_compatibility/UNIMATE_COMPATIBILITY_REPORT.md).

## How it works

```
current user ─┐
              ├─▶ user-to-flat feature builder ─▶ distance + social-fit features ─▶ XGBoost ─▶ weak / moderate / strong ─▶ score + ranking
flat residents┘
```

1. **Profiles.** Each student has a gender, age, budget range, sleep schedule, faculty, and hometown governorate. Inputs are validated with Pydantic.
2. **User-to-resident signals.** Budget difference, sleep compatibility, age difference, hometown similarity (same governorate, same region of Egypt, or neither), and faculty similarity (same faculty, same field cluster, or neither).
3. **Encoded distances.** Profiles are encoded with gender-specific transformers and feature weights; distances between the user and each resident are summarized (mean, min, max).
4. **Flat context.** The flat's own cohesion (before and after the user joins) is used as context, not as the target.
5. **XGBoost classifier** (multiclass log loss) predicts weak / moderate / strong; the probabilities give a continuous `overall_score` used to rank flats.

Top drivers: flat distance mean, combined distance mean, minimum sleep compatibility in the flat, and the user's closest resident distance.

## Training and evaluation

- `train_flat_model.py` builds a labelled dataset of flats of size 1–6 from the student pool (cohesive and random flats for each class), trains the XGBoost model with stratified validation, and saves the model, feature list, metadata, and a training data snapshot to `artifacts/user_flat_compatibility_model/`.
- `evaluate_flats.py` scores a separate holdout set (different random seed), computes metrics with a bootstrap confidence interval, draws the charts, and writes the report in `reports/user_flat_compatibility/`.

## API

### `POST /flat-scores`

```json
{
  "current_user": {
    "student_id": "demo_user_001", "gender": "Male", "age": 21, "budget_range": 0.05,
    "sleep_code": 1, "faculty": "Faculty of Computers and Artificial Intelligence",
    "hometown_governorate": "Cairo", "wants_roommate": 1
  },
  "flats": [
    {"flat_id": "demo_flat_1", "students": [
      {"student_id": "flat_member_1", "gender": "Male", "age": 22, "budget_range": 0.066,
       "sleep_code": 2, "faculty": "Faculty of Computers and Artificial Intelligence",
       "hometown_governorate": "Cairo", "wants_roommate": 1}
    ]}
  ],
  "sort_descending": true
}
```

Each scored flat returns an `overall_score`, a `compatibility_label`, class `probabilities`, explanatory `metrics` (budget, sleep, hometown and faculty fit, distances, weakest link), and notes. Interactive docs are at `/docs`.

## Run locally

```bash
git clone https://github.com/lixmed/unimate-api.git
cd unimate-api
pip install -r requirements.txt
uvicorn api:app --reload        # http://127.0.0.1:8000/docs

python train_flat_model.py      # optional: retrain
python evaluate_flats.py        # optional: regenerate the report
```

## Project structure

```
.
├── api.py                 # FastAPI app and scoring engine
├── model_utils.py         # profile canonicalization and feature extraction
├── train_flat_model.py    # dataset generation and training
├── evaluate_flats.py      # holdout evaluation and report
├── artifacts/
│   ├── unimate_matcher.joblib            # encoders, feature weights, student pools
│   └── user_flat_compatibility_model/    # XGBoost model, features, metadata, data snapshot
└── reports/user_flat_compatibility/      # metrics, charts, full report
```

## Tech stack

Python · FastAPI · Pydantic · XGBoost · scikit-learn · pandas · NumPy · SciPy · Matplotlib · Hugging Face Spaces
