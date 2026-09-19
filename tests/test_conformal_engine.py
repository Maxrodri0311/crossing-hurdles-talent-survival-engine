"""
tests/test_conformal_engine.py - Unit Test Suite for Conformalized Survival Engine
=================================================================================
Validates:
1. ConformalPredictorProtocol adherence (Clean Architecture / DIP).
2. Split-conformal calibration with finite-sample (n+1)/n quantile correction.
3. Lower bound invariants: L_i in [0.5, 16.0] and L_i <= T_50(x_i).
4. Empirical coverage mathematical guarantee: P(T_i >= L_i | delta_i = 1) >= 85% at nominal 90%.
5. Operational Runway Tier partitioning and semantic layer export.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_dataset
from src.conformal_engine import (
    ConformalPredictorProtocol,
    ConformalSurvivalEngine
)


@pytest.fixture(scope="session")
def conformal_sample_data(tmp_path_factory):
    fn = tmp_path_factory.mktemp("conformal_data") / "sample_cohort.parquet"
    df = generate_synthetic_dataset(num_records=2000, output_path=str(fn), seed=42)
    return df


def test_conformal_protocol_compliance():
    """Verifies that ConformalSurvivalEngine satisfies the abstract DIP protocol."""
    engine = ConformalSurvivalEngine()
    assert isinstance(engine, ConformalPredictorProtocol), "Engine must conform to ConformalPredictorProtocol"


def test_uncalibrated_engine_raises_error(conformal_sample_data):
    """Verifies that attempting prediction on uncalibrated engine raises RuntimeError."""
    engine = ConformalSurvivalEngine()
    with pytest.raises(RuntimeError, match="Engine has not been calibrated"):
        engine.predict_conformal_bounds(conformal_sample_data)


def test_conformal_calibration_and_finite_sample_quantile(conformal_sample_data):
    """Verifies split-conformal calibration computes a valid positive quantile correction."""
    train_df = conformal_sample_data.iloc[:1200].copy()
    cal_df = conformal_sample_data.iloc[1200:].copy()

    engine = ConformalSurvivalEngine(alpha=0.10)
    engine.fit(train_df)
    engine.calibrate(cal_df)

    assert engine._is_calibrated is True
    assert engine.conformal_quantile_correction > 0.0, "Conformal quantile correction must be strictly positive"


def test_conformal_lower_bound_invariants(conformal_sample_data):
    """Verifies mathematical invariants of the 90% conformal lower bounds."""
    train_df = conformal_sample_data.iloc[:1200].copy()
    cal_df = conformal_sample_data.iloc[1200:].copy()

    engine = ConformalSurvivalEngine(alpha=0.10)
    engine.fit(train_df)
    engine.calibrate(cal_df)

    df_bounds = engine.predict_conformal_bounds(cal_df)

    # Invariant 1: Non-negativity floor at 0.5 weeks
    assert (df_bounds["conformal_lower_bound_weeks"] >= 0.5).all(), "Lower bounds cannot be below 0.5 weeks"

    # Invariant 2: Maximum ceiling at 16.0 weeks (course length)
    assert (df_bounds["conformal_lower_bound_weeks"] <= 16.0).all(), "Lower bounds cannot exceed course horizon"

    # Invariant 3: Conservative property L_i <= T_hat_50
    assert (df_bounds["conformal_lower_bound_weeks"] <= df_bounds["predicted_median_weeks"] + 1e-5).all(), \
        "Conformal lower bound must be <= point prediction"


def test_conformal_coverage_mathematical_guarantee(conformal_sample_data):
    """Verifies that empirical coverage satisfies finite-sample theoretical guarantee."""
    train_df = conformal_sample_data.iloc[:1000].copy()
    cal_df = conformal_sample_data.iloc[1000:1500].copy()
    test_df = conformal_sample_data.iloc[1500:].copy()

    engine = ConformalSurvivalEngine(alpha=0.10)
    engine.fit(train_df)
    engine.calibrate(cal_df, alpha=0.10)

    cov_results = engine.evaluate_coverage(test_df)
    emp_cov = cov_results["empirical_coverage_pct"]

    # Nominal level 90%: finite-sample test on 500 records allows standard statistical slack (>= 85%)
    assert emp_cov >= 85.0, f"Empirical coverage {emp_cov}% violated finite-sample bound"
    assert cov_results["conformal_quantile_weeks"] > 0.0


def test_operational_runway_tiers_partitioning_and_export(conformal_sample_data, tmp_path):
    """Verifies that all candidates are assigned to valid tiers and export generates non-null marts."""
    train_df = conformal_sample_data.iloc[:1200].copy()
    cal_df = conformal_sample_data.iloc[1200:].copy()

    engine = ConformalSurvivalEngine(alpha=0.10)
    engine.fit(train_df)
    engine.calibrate(cal_df)

    out_dir = str(tmp_path / "semantic_layer")
    paths = engine.export_semantic_layer(conformal_sample_data, output_dir=out_dir)

    for key, p in paths.items():
        assert os.path.exists(p), f"Exported file {p} does not exist"

    df_tiers = pd.read_parquet(paths["conformal_tiers_parquet"])
    valid_tiers = {
        "IMMINENT_CRITICAL_WINDOW",
        "ACCELERATED_MONITORING",
        "STABLE_RUNWAY",
        "HIGH_CONFIDENCE_GRADUATION"
    }
    assert set(df_tiers["operational_runway_tier"]).issubset(valid_tiers)
    assert df_tiers["candidate_count"].sum() == len(conformal_sample_data)
