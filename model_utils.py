from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# Regional mapping for Egyptian Governorates
GOVERNORATE_REGIONS = {
    "Cairo": "Greater Cairo",
    "Giza": "Greater Cairo",
    "Qalyubia": "Greater Cairo",
    "Alexandria": "Alexandria",
    "Beheira": "Delta",
    "Dakahlia": "Delta",
    "Gharbia": "Delta",
    "Kafr El Sheikh": "Delta",
    "Monofiya": "Delta",
    "Damietta": "Delta",
    "Sharqia": "Delta",
    "Port Said": "Suez Canal",
    "Ismailia": "Suez Canal",
    "Suez": "Suez Canal",
    "Fayoum": "Upper Egypt",
    "Beni Suef": "Upper Egypt",
    "Minya": "Upper Egypt",
    "Asyut": "Upper Egypt",
    "Sohag": "Upper Egypt",
    "Qena": "Upper Egypt",
    "Luxor": "Upper Egypt",
    "Aswan": "Upper Egypt",
    "Red Sea": "Frontier",
    "New Valley": "Frontier",
    "Marsa Matrouh": "Frontier",
    "North Sinai": "Frontier",
    "South Sinai": "Frontier",
}

# Faculty groupings
FACULTY_CLUSTERS = {
    "Faculty of Computers and Artificial Intelligence": "STEM",
    "Faculty of Engineering": "STEM",
    "Faculty of Science": "STEM",
    "Faculty of Medicine": "Medical",
    "Faculty of Dentistry": "Medical",
    "Faculty of Pharmacy": "Medical",
    "Faculty of Nursing": "Medical",
    "Technical Institute of Nursing": "Medical",
    "Faculty of Arts": "Humanities",
    "Faculty of Law": "Humanities",
    "Faculty of Education": "Humanities",
    "Faculty of Tourism and Hotels": "Humanities",
    "Faculty of Social Work": "Humanities",
    "Faculty of Archaeology": "Humanities",
    "Faculty of Languages and Translation": "Humanities",
    "Faculty of Al-Alsun (Languages and Translation)": "Humanities",
    "Faculty of Physical Education": "Humanities",
    "Faculty of Agriculture": "STEM",
    "Faculty of Theology": "Humanities",
    "Institute of Nile Basin Countries": "Humanities",
    "School": "Other",
}


def get_region(gov: str) -> str:
    return GOVERNORATE_REGIONS.get(str(gov).strip(), "Other")


def get_faculty_cluster(faculty: str) -> str:
    return FACULTY_CLUSTERS.get(str(faculty).strip(), "Other")


def normalize_gender(gender: str) -> str:
    return str(gender).strip().capitalize()


def compute_hometown_similarity(gov1: str, gov2: str) -> float:
    if gov1 == gov2:
        return 1.0
    if get_region(gov1) == get_region(gov2):
        return 0.5
    return 0.0


def compute_faculty_similarity(fac1: str, fac2: str) -> float:
    if fac1 == fac2:
        return 1.0
    if get_faculty_cluster(fac1) == get_faculty_cluster(fac2):
        return 0.6
    return 0.0


def canonicalize_student(student: dict[str, Any]) -> dict[str, Any]:
    return {
        "student_id": str(student.get("student_id", "")),
        "gender": normalize_gender(student.get("gender", student.get("Gender", ""))),
        "age": int(student.get("age", student.get("Age"))),
        "budget_range": float(student.get("budget_range", student.get("Budget range"))),
        "sleep_code": int(student.get("sleep_code")),
        "faculty": str(student.get("faculty", student.get("Faculty / Study Field", ""))).strip(),
        "hometown_governorate": str(
            student.get("hometown_governorate", student.get("Hometown Governorate", ""))
        ).strip(),
        "wants_roommate": int(student.get("wants_roommate", 1)),
    }


def students_to_transformer_frame(students: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Age": student["age"],
            "Budget range": student["budget_range"],
            "sleep_code": student["sleep_code"],
            "Faculty / Study Field": student["faculty"],
            "Hometown Governorate": student["hometown_governorate"],
        }
        for student in students
    ]
    return pd.DataFrame(rows)


def _safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_pair_metrics(students: list[dict[str, Any]], max_budget_diff: float) -> dict[str, float]:
    budget_diffs: list[float] = []
    sleep_compats: list[float] = []
    age_diffs: list[float] = []
    hometown_sims: list[float] = []
    faculty_sims: list[float] = []

    for s1, s2 in combinations(students, 2):
        budget_diffs.append(abs(s1["budget_range"] - s2["budget_range"]))
        sleep_compats.append(1 - abs(s1["sleep_code"] - s2["sleep_code"]) / 2)
        age_diffs.append(abs(s1["age"] - s2["age"]))
        hometown_sims.append(
            compute_hometown_similarity(s1["hometown_governorate"], s2["hometown_governorate"])
        )
        faculty_sims.append(compute_faculty_similarity(s1["faculty"], s2["faculty"]))

    budget_stats = _safe_stats(budget_diffs)
    sleep_stats = _safe_stats(sleep_compats)
    age_stats = _safe_stats(age_diffs)
    hometown_stats = _safe_stats(hometown_sims)
    faculty_stats = _safe_stats(faculty_sims)
    pass_rate = float(np.mean([1.0 if diff <= max_budget_diff else 0.0 for diff in budget_diffs])) if budget_diffs else 1.0

    return {
        "pair_count": float(len(budget_diffs)),
        "budget_diff_mean": budget_stats["mean"],
        "budget_diff_max": budget_stats["max"],
        "sleep_compat_mean": sleep_stats["mean"],
        "sleep_compat_min": sleep_stats["min"],
        "age_diff_mean": age_stats["mean"],
        "age_diff_max": age_stats["max"],
        "hometown_sim_mean": hometown_stats["mean"],
        "faculty_sim_mean": faculty_stats["mean"],
        "budget_pass_rate": pass_rate,
    }


def summarize_distance_values(values: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def extract_flat_features(students: list[dict[str, Any]], max_budget_diff: float) -> dict[str, float]:
    normalized = [canonicalize_student(student) for student in students]
    pair_metrics = _safe_pair_metrics(normalized, max_budget_diff)
    ages = [student["age"] for student in normalized]
    return {
        "avg_budget_diff": pair_metrics["budget_diff_mean"],
        "avg_sleep_compat": pair_metrics["sleep_compat_mean"],
        "avg_age_diff": pair_metrics["age_diff_mean"],
        "avg_hometown_sim": pair_metrics["hometown_sim_mean"],
        "avg_faculty_sim": pair_metrics["faculty_sim_mean"],
        "budget_sleep_interaction": pair_metrics["budget_diff_mean"] * (1 - pair_metrics["sleep_compat_mean"]),
        "budget_pass_rate": pair_metrics["budget_pass_rate"],
        "student_count": float(len(normalized)),
        "age_span": float(max(ages) - min(ages)) if ages else 0.0,
    }


def extract_user_flat_features(
    current_user: dict[str, Any],
    flat_students: list[dict[str, Any]],
    max_budget_diff: float,
) -> dict[str, float]:
    user = canonicalize_student(current_user)
    occupants = [canonicalize_student(student) for student in flat_students]
    combined = [user, *occupants]

    user_budget_diffs = [abs(user["budget_range"] - occupant["budget_range"]) for occupant in occupants]
    user_sleep_compats = [1 - abs(user["sleep_code"] - occupant["sleep_code"]) / 2 for occupant in occupants]
    user_age_diffs = [abs(user["age"] - occupant["age"]) for occupant in occupants]
    user_hometown_sims = [
        compute_hometown_similarity(user["hometown_governorate"], occupant["hometown_governorate"])
        for occupant in occupants
    ]
    user_faculty_sims = [
        compute_faculty_similarity(user["faculty"], occupant["faculty"])
        for occupant in occupants
    ]

    user_budget_stats = _safe_stats(user_budget_diffs)
    user_sleep_stats = _safe_stats(user_sleep_compats)
    user_age_stats = _safe_stats(user_age_diffs)
    user_hometown_stats = _safe_stats(user_hometown_sims)
    user_faculty_stats = _safe_stats(user_faculty_sims)

    occupant_metrics = _safe_pair_metrics(occupants, max_budget_diff)
    combined_metrics = _safe_pair_metrics(combined, max_budget_diff)
    occupant_ages = [student["age"] for student in occupants]
    combined_ages = [student["age"] for student in combined]

    return {
        "flat_size": float(len(occupants)),
        "flat_pair_count": occupant_metrics["pair_count"],
        "user_budget_diff_mean": user_budget_stats["mean"],
        "user_budget_diff_max": user_budget_stats["max"],
        "user_sleep_compat_mean": user_sleep_stats["mean"],
        "user_sleep_compat_min": user_sleep_stats["min"],
        "user_age_diff_mean": user_age_stats["mean"],
        "user_age_diff_max": user_age_stats["max"],
        "user_hometown_sim_mean": user_hometown_stats["mean"],
        "user_faculty_sim_mean": user_faculty_stats["mean"],
        "user_budget_pass_rate": float(
            np.mean([1.0 if diff <= max_budget_diff else 0.0 for diff in user_budget_diffs]) if user_budget_diffs else 1.0
        ),
        "flat_budget_diff_mean": occupant_metrics["budget_diff_mean"],
        "flat_budget_diff_max": occupant_metrics["budget_diff_max"],
        "flat_sleep_compat_mean": occupant_metrics["sleep_compat_mean"],
        "flat_sleep_compat_min": occupant_metrics["sleep_compat_min"],
        "flat_age_diff_mean": occupant_metrics["age_diff_mean"],
        "flat_hometown_sim_mean": occupant_metrics["hometown_sim_mean"],
        "flat_faculty_sim_mean": occupant_metrics["faculty_sim_mean"],
        "flat_budget_pass_rate": occupant_metrics["budget_pass_rate"],
        "flat_age_span": float(max(occupant_ages) - min(occupant_ages)) if occupant_ages else 0.0,
        "combined_budget_diff_mean": combined_metrics["budget_diff_mean"],
        "combined_budget_diff_max": combined_metrics["budget_diff_max"],
        "combined_sleep_compat_mean": combined_metrics["sleep_compat_mean"],
        "combined_sleep_compat_min": combined_metrics["sleep_compat_min"],
        "combined_age_diff_mean": combined_metrics["age_diff_mean"],
        "combined_hometown_sim_mean": combined_metrics["hometown_sim_mean"],
        "combined_faculty_sim_mean": combined_metrics["faculty_sim_mean"],
        "combined_budget_pass_rate": combined_metrics["budget_pass_rate"],
        "combined_age_span": float(max(combined_ages) - min(combined_ages)) if combined_ages else 0.0,
        "delta_budget_diff_mean": combined_metrics["budget_diff_mean"] - occupant_metrics["budget_diff_mean"],
        "delta_sleep_compat_mean": combined_metrics["sleep_compat_mean"] - occupant_metrics["sleep_compat_mean"],
        "delta_hometown_sim_mean": combined_metrics["hometown_sim_mean"] - occupant_metrics["hometown_sim_mean"],
        "delta_faculty_sim_mean": combined_metrics["faculty_sim_mean"] - occupant_metrics["faculty_sim_mean"],
    }
