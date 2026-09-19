"""
tests/test_suite.py - Mathematical Invariants & Architecture Verification
==========================================================================
Validates:
1. Data contracts and schema invariants.
2. Kaplan-Meier mathematical invariants (monotonicity, bounds, CI coverage).
3. Actuarial life table population conservation.
4. Explainable Hazard Ratios statistical validity (HR > 1.0, CI ordering).
5. Clean Architecture / Dependency Inversion Principle (DIP) mock isolation.
6. Execution latency (<30ms).
"""

import os
import sys
import tempfile
import time
import pytest
import pandas as pd
import numpy as np

from src.data_generator import generate_synthetic_dataset
from src.core_engine import (
    AnalyticsEngine,
    DuckDBStorageAdapter,
    AnalyticalStorageProtocol,
    create_engine
)


@pytest.fixture(scope="session")
def test_dataset(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_data.parquet"
    df = generate_synthetic_dataset(num_records=10000, output_path=str(fn), seed=42)
    return str(fn)


def test_data_generation_contracts(test_dataset):
    """Verifies schema, boundary constraints, and non-empty columns."""
    df = pd.read_parquet(test_dataset)
    assert len(df) == 10000
    expected_cols = [
        "candidate_id", "cohort_id", "track", "prior_experience",
        "weekly_hours_dedicated", "assignment_lag_days", "tutor_feedback_latency_hours",
        "hurdle_difficulty_index", "assessment_score_avg", "duration_weeks",
        "event_observed", "lifecycle_status", "primary_drop_reason"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert df["duration_weeks"].min() >= 0.5
    assert df["duration_weeks"].max() <= 16.0
    assert set(df["event_observed"].unique()).issubset({0, 1})
    assert df["event_observed"].sum() > 0


def test_kaplan_meier_mathematical_invariants(test_dataset):
    """
    Verifies fundamental mathematical laws of Kaplan-Meier:
    - Survival S(t) is strictly monotonically non-increasing.
    - S(t) in [0, 1] for all t.
    - S(0) close to 1.0.
    - Confidence Interval: lower <= S(t) <= upper.
    """
    engine = create_engine(data_path=test_dataset)
    km_df = engine.compute_kaplan_meier_overall()
    assert not km_df.empty

    # Initial time survival
    assert km_df.iloc[0]["survival_probability"] <= 1.0
    assert km_df.iloc[0]["survival_probability"] >= 0.95

    # Monotonicity check
    s_probs = km_df["survival_probability"].values
    for i in range(len(s_probs) - 1):
        assert s_probs[i + 1] <= s_probs[i] + 1e-6, f"Monotonicity violated at step {i}: {s_probs[i]} -> {s_probs[i+1]}"

    # Bounds check
    assert (km_df["survival_probability"] >= 0.0).all()
    assert (km_df["survival_probability"] <= 1.0).all()

    # CI checks
    assert (km_df["ci_95_lower"] <= km_df["survival_probability"] + 1e-4).all()
    assert (km_df["ci_95_upper"] >= km_df["survival_probability"] - 1e-4).all()


def test_actuarial_life_table_population_conservation(test_dataset):
    """
    Verifies that life table preserves total population and conditional rates:
    q_x + p_x == 1.0.
    """
    engine = create_engine(data_path=test_dataset)
    life_df = engine.compute_actuarial_life_table()
    assert not life_df.empty

    # Conservation of individuals
    total_events = life_df["dropouts_observed"].sum() + life_df["censored_active"].sum()
    assert total_events == 10000

    # Rate sum invariant
    for _, row in life_df.iterrows():
        qx = row["conditional_dropout_rate_qx"]
        px = row["conditional_survival_rate_px"]
        assert abs((qx + px) - 1.0) < 1e-4


def test_explainable_hazard_ratios_statistical_validity(test_dataset):
    """
    Verifies that all 5 critical learning barriers yield statistically valid HRs:
    HR > 1.0 and CI lower < HR < CI upper.
    """
    engine = create_engine(data_path=test_dataset)
    hr_df = engine.compute_explainable_hazard_ratios()
    assert len(hr_df) == 5

    for _, row in hr_df.iterrows():
        hr = row["hazard_ratio"]
        low = row["ci_95_lower"]
        high = row["ci_95_upper"]
        assert hr >= 1.0, f"Hazard ratio should indicate increased risk: {row['learning_barrier']}"
        assert low <= hr <= high, f"CI malformed for {row['learning_barrier']}"
        assert len(row["executive_interpretation"]) > 20
        assert len(row["recommended_business_action"]) > 20


def test_clean_architecture_dip_mock_isolation():
    """
    Demonstrates true Dependency Inversion Principle (DIP):
    Domain engine functions with an in-memory mock adapter without touching disk or DuckDB.
    """
    class InMemoryMockStorage:
        def execute_query(self, query: str) -> pd.DataFrame:
            if "total_pop" in query:
                return pd.DataFrame([{"total_pop": 1000}])
            if "timeline" in query:
                return pd.DataFrame([
                    {"time_week": 1.0, "events_count": 50, "censored_count": 10, "total_records": 60},
                    {"time_week": 2.0, "events_count": 40, "censored_count": 15, "total_records": 55}
                ])
            return pd.DataFrame([{"total_pop": 1000}])

    mock_storage = InMemoryMockStorage()
    engine = AnalyticsEngine(storage=mock_storage, data_path="dummy_path.parquet")
    
    # Bypass file existence for pure in-memory test
    engine._ensure_data_exists = lambda: None
    km = engine.compute_kaplan_meier_overall()
    assert len(km) == 2
    assert km.iloc[0]["n_at_risk"] == 1000


def test_query_latency_benchmark_sub30ms(test_dataset):
    """Verifies that DuckDB execution over 10,000 records takes < 30ms."""
    engine = create_engine(data_path=test_dataset)
    start = time.perf_counter()
    engine.compute_actuarial_life_table()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 30.0, f"Query latency exceeded 30ms SLA: {elapsed_ms:.2f}ms"
