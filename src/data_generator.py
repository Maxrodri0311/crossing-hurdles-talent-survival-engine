"""
src/data_generator.py - Stochastic Data Generator for Crossing Hurdles
======================================================================
Generates 50,000+ synthetic student/talent records modeling authentic EdTech
career acceleration and learning barrier drop-off dynamics:
- Pacing overload vs assignment lag
- Tutor feedback latency impact on attrition hazard
- Right-censored student trajectories (active vs graduated vs dropped)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Windows UTF-8 physics
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def generate_synthetic_dataset(
    num_records: int = 50000,
    output_path: str = "data/raw_dataset.parquet",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic EdTech career acceleration telemetry with stochastic
    survival kinetics and causal covariates.
    """
    print(f"[Data Generator] Generating {num_records:,} student cohort records for Crossing Hurdles...")
    start_time = time.time()
    np.random.seed(seed)

    candidate_ids = np.arange(100001, 100001 + num_records)
    
    # 1. Cohorts & Tracks
    cohorts = ["Cohort_2025_Q1", "Cohort_2025_Q2", "Cohort_2025_Q3", "Cohort_2025_Q4"]
    tracks = ["Data Science", "Machine Learning Ops", "Full-Stack AI", "Data Analytics"]
    backgrounds = [
        "Non-Technical Career Switcher",
        "Self-Taught Coder",
        "Junior Developer",
        "STEM University Graduate"
    ]
    
    cohort_col = np.random.choice(cohorts, size=num_records, p=[0.28, 0.26, 0.24, 0.22])
    track_col = np.random.choice(tracks, size=num_records, p=[0.35, 0.25, 0.20, 0.20])
    background_col = np.random.choice(backgrounds, size=num_records, p=[0.30, 0.30, 0.25, 0.15])
    
    # 2. Continuous Covariates with realistic covariance
    base_hours = np.random.normal(loc=20.0, scale=6.0, size=num_records)
    switcher_mask = (background_col == "Non-Technical Career Switcher")
    base_hours[switcher_mask] -= 4.5
    weekly_hours = np.clip(np.round(base_hours, 1), 5.0, 45.0)

    # Assignment submission lag in days (mean 2.5 days)
    base_lag = np.random.exponential(scale=2.2, size=num_records) + 0.5
    base_lag[switcher_mask] += 1.2
    assignment_lag_days = np.clip(np.round(base_lag, 1), 0.5, 14.0)

    # Tutor feedback latency in hours (mean ~15 hrs)
    tutor_latency_hours = np.clip(np.round(np.random.exponential(scale=14.0, size=num_records) + 2.0, 1), 1.0, 96.0)

    # Hurdle difficulty index (1.0 to 5.0 scale)
    hurdle_difficulty = np.round(np.random.uniform(1.5, 4.8, size=num_records), 2)

    # Assessment score average (0 to 100)
    score_mean = 78.0 - (assignment_lag_days * 1.6) - (hurdle_difficulty * 3.2) + (weekly_hours * 0.35)
    assessment_score = np.clip(np.round(np.random.normal(loc=score_mean, scale=8.0), 1), 30.0, 100.0)

    # 3. Survival Hazard & Event Kinetics
    risk_score = (
        -0.08 * (weekly_hours - 18.0)
        + 0.18 * (assignment_lag_days - 2.5)
        + 0.03 * (tutor_latency_hours - 14.0)
        + 0.40 * (hurdle_difficulty - 3.0)
        - 0.04 * (assessment_score - 72.0)
    )
    # Background baseline risk adjustment
    risk_score += np.where(background_col == "Non-Technical Career Switcher", 0.45, 0.0)
    risk_score -= np.where(background_col == "STEM University Graduate", 0.50, 0.0)

    # Logistic event probability before graduation
    p_dropout = 1.0 / (1.0 + np.exp(-(risk_score - 0.20)))
    event_observed = (np.random.rand(num_records) < p_dropout).astype(int)

    # For dropouts: event time distributed across weeks 1 to 15 (peak weeks 3-7)
    dropout_times = np.random.weibull(a=1.65, size=num_records) * 5.4 + 0.8
    dropout_times = np.clip(np.round(dropout_times, 2), 0.5, 15.2)

    # For non-dropouts: Reaching 16.0 weeks (graduated) or right-censored in Q4 active cohort
    is_recent_cohort = (cohort_col == "Cohort_2025_Q4")
    active_censor_time = np.random.uniform(4.0, 14.0, size=num_records)

    duration_weeks = np.where(
        event_observed == 1,
        dropout_times,
        np.where(is_recent_cohort, np.round(active_censor_time, 2), 16.00)
    )

    # Categorize status & primary reason
    reasons = []
    status_list = []
    for i in range(num_records):
        obs = event_observed[i]
        dur = duration_weeks[i]
        if obs == 0:
            if dur >= 15.9:
                status_list.append("GRADUATED")
                reasons.append("Completed Program")
            else:
                status_list.append("ACTIVE_IN_COURSE")
                reasons.append("In Progress (Censored)")
        else:
            status_list.append("DROPPED_OUT")
            if tutor_latency_hours[i] > 36.0:
                reasons.append("Lack of Mentor Support")
            elif weekly_hours[i] < 12.0:
                reasons.append("Pacing Overwhelm / Time Deficit")
            elif assessment_score[i] < 60.0:
                reasons.append("Technical Hurdle Failure")
            elif assignment_lag_days[i] > 5.0:
                reasons.append("Milestone Lag Disengagement")
            else:
                reasons.append("Financial / Personal Factors")

    # Build DataFrame
    df = pd.DataFrame({
        "candidate_id": candidate_ids,
        "cohort_id": cohort_col,
        "track": track_col,
        "prior_experience": background_col,
        "weekly_hours_dedicated": weekly_hours,
        "assignment_lag_days": assignment_lag_days,
        "tutor_feedback_latency_hours": tutor_latency_hours,
        "hurdle_difficulty_index": hurdle_difficulty,
        "assessment_score_avg": assessment_score,
        "duration_weeks": np.round(duration_weeks, 2),
        "event_observed": event_observed,
        "lifecycle_status": status_list,
        "primary_drop_reason": reasons
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    elapsed = time.time() - start_time
    total_dropouts = int(df["event_observed"].sum())
    dropout_rate = (total_dropouts / num_records) * 100.0
    print(f"[Data Generator] Generated {len(df):,} records in {elapsed:.2f}s -> {output_path}")
    print(f"[Data Generator] Observed Dropouts: {total_dropouts:,} ({dropout_rate:.1f}%) | Censored/Active: {num_records - total_dropouts:,}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossing Hurdles Synthetic Data Generator")
    parser.add_argument("--records", type=int, default=50000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/raw_dataset.parquet", help="Output path")
    args = parser.parse_args()
    generate_synthetic_dataset(args.records, args.output)
