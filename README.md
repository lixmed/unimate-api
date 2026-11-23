# UniMate — Flat Compatibility Scoring API

UniMate is an AI-powered student housing platform. This repository is its **scoring API**: given the students who would share a flat, it predicts how well they would live together and returns a compatibility label, a score from 0 to 1, and the reasons behind it.

Built with **FastAPI**, **XGBoost**, and **scikit-learn**.

## Results

| Metric | Train | Test |
|---|---|---|
| Accuracy | 98.87% | **98.37%** |
| F1 score | 0.989 | **0.984** |

Train/test gap: 0.5%, so the model generalizes instead of memorizing.

## How it works

1. **Student profiles.** Each student has a gender, age, budget range, sleep schedule, faculty, hometown governorate, and whether they want a roommate. Inputs are validated with Pydantic (for example, age 17–40, budget 0–1).
2. **Pairwise compatibility.** For every pair of students in the flat, the API computes:
   - budget difference, and whether the pair is within the budget threshold
   - sleep schedule compatibility
   - age difference
   - hometown similarity: same governorate, same region of Egypt (e.g. Greater Cairo, Delta, Upper Egypt), or neither
   - faculty similarity: same faculty, same field cluster (STEM, Medical, Humanities), or neither
3. **Weighted distance features.** Profiles are encoded with gender-specific transformers and feature weights; pairwise Euclidean distances give the average, closest, and weakest pair in the flat.
4. **Flat-level features.** Pair features are aggregated into 14 flat-level features (averages, age span, budget–sleep interaction, a penalty for the weakest pair).
5. **XGBoost classifier.** Predicts **weak / moderate / strong** compatibility. The class probabilities are also combined into a continuous `overall_score` (0–1) so flats can be ranked.

## API

### `POST /flat-scores`

Scores up to **500 flats** in one request, each with up to **20 students**. Flats are sorted by score (best first) by default. Mixed-gender flats are returned under `rejected_flats` with a reason.

**Request**

```json
{
  "flats": [
    {
      "flat_id": "flat-1",
      "students": [
        {"student_id": "s1", "gender": "Male", "age": 20, "budget_range": 0.4, "sleep_code": 1,
         "faculty": "Faculty of Computers and Artificial Intelligence", "hometown_governorate": "Fayoum"},
        {"student_id": "s2", "gender": "Male", "age": 21, "budget_range": 0.5, "sleep_code": 1,
         "faculty": "Faculty of Engineering", "hometown_governorate": "Beni Suef"}
      ]
    }
  ],
  "sort_descending": true
}
```

**Response** (shape)

```json
{
  "request_count": 1,
  "scored_count": 1,
  "rejected_count": 0,
  "flat_scores": [
    {
      "flat_id": "flat-1",
      "overall_score": 0.0,
      "compatibility_label": "strong | moderate | weak",
      "probabilities": {"weak": 0.0, "moderate": 0.0, "strong": 0.0},
      "metrics": {
        "avg_budget_diff": 0.0,
        "avg_sleep_compatibility": 0.0,
        "avg_hometown_similarity": 0.0,
        "avg_faculty_similarity": 0.0,
        "weakest_pair_score": 0.0,
        "all_pairs_within_budget_threshold": true
      }
    }
  ],
  "rejected_flats": []
}
```

Interactive docs are available at `/docs` once the server is running.

## Run locally

```bash
git clone https://github.com/lixmed/unimate-api.git
cd unimate-api
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000/docs.

## Project structure

```
.
├── api.py               # FastAPI app, request validation, scoring engine
├── model_utils.py       # pairwise feature extraction (budget, sleep, age, hometown, faculty)
├── requirements.txt
└── artifacts/
    ├── unimate_matcher.joblib     # encoders and gender-specific feature weights
    └── flat_model/
        ├── xgb_model.joblib       # trained XGBoost classifier
        ├── feature_names.json     # feature order expected by the model
        ├── thresholds.json        # score thresholds for moderate / strong
        └── robustness_metrics.json
```

## Tech stack

Python · FastAPI · Pydantic · XGBoost · scikit-learn · pandas · NumPy · SciPy · joblib
