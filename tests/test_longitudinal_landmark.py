"""
tests/test_longitudinal_landmark.py - Unit Test Suite for Landmark Survival Engine
==================================================================================
Validates:
1. Strict landmark conditioning: only active students (duration_weeks > t_L) enter cohort.
2. Horizon truncation math: t_residual in (0, horizon] and event_in_horizon definition.
3. Dynamic C-index discrimination progression across landmarks (C > 0.75).
4. Operational alert risk tier partitioning and 100% prescriptive action coverage.
5. Clean Architecture (DIP Protocol compliance).
6. Semantic layer mart schema integrity and non-null guarantees.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_dataset
from src.longitudinal_engine import (
    LandmarkEngineProtocol,
    LandmarkTrajectoryTransformer,
    DynamicLandmarkEngine
)


@pytest.fixture(scope="session")
def landmark_sample_data(tmp_path_factory):
    fn = tmp_path_factory.mktemp("landmark_data") / "sample_cohort.parquet"
    df = generate_synthetic_dataset(num_records=3000, output_path=str(fn), seed=2026)
    return df


def test_landmark_cohort_filtering_and_invariants(landmark_sample_data):
    """Verifies that landmark cohort extraction enforces temporal eligibility and horizon math."""
    t_L = 3.0
    horizon = 4.0
    cohort = LandmarkTrajectoryTransformer.extract_landmark_cohort(
        landmark_sample_data, t_landmark=t_L, horizon=horizon, seed=42
    )

    # Invariant 1: All candidates in cohort must have survived beyond t_L
    assert (cohort["duration_weeks"] > t_L).all(), "Temporal violation: inactive candidate in landmark cohort"

    # Invariant 2: Residual duration must be bounded within (0, horizon]
    assert (cohort["t_residual"] > 0.0).all()
    assert (cohort["t_residual"] <= horizon + 1e-5).all()

    # Invariant 3: Event in horizon matches true status
    for _, row in cohort.iterrows():
        if row["duration_weeks"] <= t_L + horizon and row["event_observed"] == 1:
            assert row["event_in_horizon"] == 1
        else:
            assert row["event_in_horizon"] == 0

    # Invariant 4: Longitudinal momentum features are non-null
    assert cohort["hours_decay_slope"].isna().sum() == 0
    assert cohort["lag_acceleration"].isna().sum() == 0


def test_dynamic_c_index_progression(landmark_sample_data):
    """Verifies that dynamic landmark models achieve high discrimination power (C > 0.75)."""
    engine = DynamicLandmarkEngine(landmark_times=[3.0, 5.0], horizon=4.0, penalizer=0.01)
    engine.fit(landmark_sample_data)

    eval_df = engine.evaluate()
    assert len(eval_df) == 2

    for _, row in eval_df.iterrows():
        c_idx = row["concordance_index_c"]
        assert 0.50 <= c_idx <= 1.0, f"C-index out of bounds: {c_idx}"
        assert c_idx >= 0.75, f"Dynamic C-index at Week {row['landmark_week']} ({c_idx}) below expected threshold"


def test_operational_alert_tiers_and_actions(landmark_sample_data):
    """Verifies alert tier partitioning, root cause diagnostics, and non-empty actions."""
    engine = DynamicLandmarkEngine(landmark_times=[3.0], horizon=4.0, penalizer=0.01)
    engine.fit(landmark_sample_data)

    df_alerts = engine.generate_alerts(landmark_sample_data, alert_percentile=85.0)
    assert not df_alerts.empty
    assert set(df_alerts["risk_tier"].unique()).issubset({"CRITICAL_ALERT", "HIGH_RISK", "MODERATE_RISK", "STABLE"})

    # Check non-null prescriptive action guarantees
    assert df_alerts["prescribed_coordinator_action"].isna().sum() == 0
    assert (df_alerts["prescribed_coordinator_action"].str.len() > 10).all()

    # Verify critical alerts have dynamic hazard scores strictly above median
    median_hazard = df_alerts["dynamic_hazard_score"].median()
    critical_scores = df_alerts[df_alerts["risk_tier"] == "CRITICAL_ALERT"]["dynamic_hazard_score"]
    assert (critical_scores >= median_hazard).all()


def test_landmark_dip_protocol_compliance():
    """Verifies that DynamicLandmarkEngine conforms to LandmarkEngineProtocol."""
    engine = DynamicLandmarkEngine()
    assert isinstance(engine, LandmarkEngineProtocol), "Engine violates DIP LandmarkEngineProtocol"


def test_semantic_layer_landmark_marts_export(landmark_sample_data, tmp_path):
    """Verifies export of landmark dimension and operational alerts fact table."""
    engine = DynamicLandmarkEngine(landmark_times=[3.0, 5.0], horizon=4.0, penalizer=0.01)
    engine.fit(landmark_sample_data)

    output_dir = str(tmp_path / "semantic_layer")
    paths = engine.export_semantic_layer(landmark_sample_data, output_dir=output_dir)

    for key, path in paths.items():
        assert os.path.exists(path), f"Landmark mart file missing: {path}"
        assert os.path.getsize(path) > 0, f"Empty file: {path}"

    df_eval = pd.read_parquet(paths["landmark_eval_parquet"])
    assert "concordance_index_c" in df_eval.columns
    assert "event_rate_pct" in df_eval.columns
    assert len(df_eval) == 2

    df_alerts = pd.read_parquet(paths["landmark_alerts_parquet"])
    assert "risk_tier" in df_alerts.columns
    assert "prescribed_coordinator_action" in df_alerts.columns
    assert len(df_alerts) > 0
