from __future__ import annotations
import numpy as np
import pandas as pd

# Regional mapping for Egyptian Governorates
GOVERNORATE_REGIONS = {
    'Cairo': 'Greater Cairo', 'Giza': 'Greater Cairo', 'Qalyubia': 'Greater Cairo',
    'Alexandria': 'Alexandria',
    'Beheira': 'Delta', 'Dakahlia': 'Delta', 'Gharbia': 'Delta', 'Kafr El Sheikh': 'Delta', 'Monofiya': 'Delta', 'Damietta': 'Delta', 'Sharqia': 'Delta', 'Port Said': 'Suez Canal', 'Ismailia': 'Suez Canal', 'Suez': 'Suez Canal',
    'Fayoum': 'Upper Egypt', 'Beni Suef': 'Upper Egypt', 'Minya': 'Upper Egypt', 'Asyut': 'Upper Egypt', 'Sohag': 'Upper Egypt', 'Qena': 'Upper Egypt', 'Luxor': 'Upper Egypt', 'Aswan': 'Upper Egypt',
    'Red Sea': 'Frontier', 'New Valley': 'Frontier', 'Marsa Matrouh': 'Frontier', 'North Sinai': 'Frontier', 'South Sinai': 'Frontier'
}

# Faculty groupings
FACULTY_CLUSTERS = {
    'Faculty of Computers and Artificial Intelligence': 'STEM',
    'Faculty of Engineering': 'STEM',
    'Faculty of Science': 'STEM',
    'Faculty of Medicine': 'Medical',
    'Faculty of Dentistry': 'Medical',
    'Faculty of Pharmacy': 'Medical',
    'Faculty of Nursing': 'Medical',
    'Technical Institute of Nursing': 'Medical',
    'Faculty of Arts': 'Humanities',
    'Faculty of Law': 'Humanities',
    'Faculty of Education': 'Humanities',
    'Faculty of Tourism and Hotels': 'Humanities',
    'Faculty of Social Work': 'Humanities',
    'Faculty of Archaeology': 'Humanities',
    'Faculty of Languages and Translation': 'Humanities',
    'Faculty of Al-Alsun (Languages and Translation)': 'Humanities',
    'Faculty of Physical Education': 'Humanities',
    'Faculty of Agriculture': 'STEM',
    'Faculty of Theology': 'Humanities',
    'Institute of Nile Basin Countries': 'Humanities',
    'School': 'Other'
}

def get_region(gov: str) -> str:
    return GOVERNORATE_REGIONS.get(gov, 'Other')

def get_faculty_cluster(faculty: str) -> str:
    return FACULTY_CLUSTERS.get(faculty, 'Other')

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

def extract_flat_features(students: list[dict[str, any]], max_budget_diff: float) -> dict[str, float]:
    """
    Extracts high-level features from a list of student profiles for a flat.
    """
    if len(students) < 2:
        return {}
        
    pair_distances = []
    budget_diffs = []
    sleep_compats = []
    age_diffs = []
    hometown_sims = []
    faculty_sims = []
    
    from itertools import combinations
    for s1, s2 in combinations(students, 2):
        # Basic diffs
        b_diff = abs(s1['budget_range'] - s2['budget_range'])
        s_compat = 1 - abs(s1['sleep_code'] - s2['sleep_code']) / 2
        a_diff = abs(s1['age'] - s2['age'])
        
        budget_diffs.append(b_diff)
        sleep_compats.append(s_compat)
        age_diffs.append(a_diff)
        
        # New similarity features
        hometown_sims.append(compute_hometown_similarity(s1['hometown_governorate'], s2['hometown_governorate']))
        faculty_sims.append(compute_faculty_similarity(s1['faculty'], s2['faculty']))

    # Aggregate features
    avg_budget_diff = np.mean(budget_diffs)
    avg_sleep_compat = np.mean(sleep_compats)
    avg_age_diff = np.mean(age_diffs)
    avg_hometown_sim = np.mean(hometown_sims)
    avg_faculty_sim = np.mean(faculty_sims)
    
    # Interaction features
    budget_sleep_interaction = avg_budget_diff * (1 - avg_sleep_compat)
    
    # Budget threshold features
    budget_pass_rate = np.mean([1.0 if d <= max_budget_diff else 0.0 for d in budget_diffs])
    
    # Penalty features
    # We'll use a placeholder for now since we don't have the distance yet in this function
    # The actual XGBoost model will take these as inputs
    
    return {
        'avg_budget_diff': float(avg_budget_diff),
        'avg_sleep_compat': float(avg_sleep_compat),
        'avg_age_diff': float(avg_age_diff),
        'avg_hometown_sim': float(avg_hometown_sim),
        'avg_faculty_sim': float(avg_faculty_sim),
        'budget_sleep_interaction': float(budget_sleep_interaction),
        'budget_pass_rate': float(budget_pass_rate),
        'student_count': float(len(students)),
        'age_span': float(max(s['age'] for s in students) - min(s['age'] for s in students))
    }
