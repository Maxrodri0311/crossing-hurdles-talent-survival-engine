"""
tests/test_survival_ml.py - Verification Suite for Survival ML & Calibration Engine
===================================================================================
Validates:
1. Feature Transformer contracts and high-order interaction math.
2. Regularized Multivariate Cox convergence, C-Index bounds (0.5 <= C <= 1.0).
3. Integrated Brier Score (IBS) probabilistic calibration (<0.25).
4. Individual survival curve monotonicity (S(t1) >= S(t2) for t1 <= t2).
5. Clean Architecture (DIP Protocol compliance).
6. Semantic layer mart schema integrity and non-null guarantees.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_dataset
from src.survival_ml import (
    SurvivalPredictorProtocol,
    SurvivalFeatureTransformer,
    MultivariateSurvivalMLEngine
)


@pytest.fixture(scope="session")
def sample_ml_data(tmp_path_factory):
    fn = tmp_path_factory.mktemp("ml_data") / "sample_students.parquet"
    df = generate_synthetic_dataset(num_records=2500, output_path=str(fn), seed=1337)
    return df


def test_feature_transformer_invariants(sample_ml_data):
    """Verifies feature transformer produces exact schemas, binary indicators, and no NaNs."""
    df_feat = SurvivalFeatureTransformer.transform(sample_ml_data, include_targets=True)

    expected_cols = [
        "weekly_hours_dedicated", "assignment_lag_days", "tutor_feedback_latency_hours",
        "hurdle_difficulty_index", "assessment_score_avg", "is_career_switcher",
        "is_stem_graduate", "interaction_switcher_latency", "interaction_switcher_difficulty",
        "interaction_hours_deficit_lag", "duration_weeks", "event_observed"
    ]
    for col in expected_cols:
        assert col in df_feat.columns, f"Missing feature column: {col}"
        assert df_feat[col].isna().sum() == 0, f"NaNs detected in feature: {col}"

    assert set(df_feat["is_career_switcher"].unique()).issubset({0.0, 1.0})
    assert set(df_feat["is_stem_graduate"].unique()).issubset({0.0, 1.0})
    assert (df_feat["interaction_switcher_latency"] >= 0.0).all()
    assert (df_feat["interaction_hours_deficit_lag"] >= 0.0).all()


def test_multivariate_cox_convergence_and_metrics(sample_ml_data):
    """Verifies model convergence, C-Index discrimination, and Brier score calibration."""
    engine = MultivariateSurvivalMLEngine(penalizer=0.01)
    engine.fit(sample_ml_data)

    c_index = engine.compute_concordance_index(sample_ml_data)
    assert 0.50 <= c_index <= 1.0, f"C-index out of theoretical bounds: {c_index}"
    assert c_index > 0.60, f"C-index {c_index} indicates insufficient discrimination power"

    ibs, brier_curve = engine.compute_integrated_brier_score(sample_ml_data, eval_times=[4.0, 8.0, 12.0])
    assert 0.0 <= ibs <= 0.25, f"Integrated Brier Score {ibs} exceeded benchmark threshold of 0.25"

    for t, bs in brier_curve.items():
        assert 0.0 <= bs <= 0.35, f"Brier score at week {t} ({bs}) exceeded threshold"


def test_survival_predictions_monotonicity_and_bounds(sample_ml_data):
    """Verifies predicted survival probabilities are strictly in [0, 1] and monotonic over time."""
    engine = MultivariateSurvivalMLEngine(penalizer=0.01)
    engine.fit(sample_ml_data)

    subset = sample_ml_data.head(50)
    times = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    surv_matrix = engine.predict_survival_matrix(subset, times=times)

    # Matrix dimensions: len(times) rows x len(subset) columns
    assert surv_matrix.shape[0] == len(times)
    assert surv_matrix.shape[1] == len(subset)

    # Invariants check
    for col in surv_matrix.columns:
        curve = surv_matrix[col].to_numpy()
        assert (curve >= 0.0).all() and (curve <= 1.0).all(), "Survival probability out of [0, 1]"
        # S(t) must be non-increasing over time
        diffs = np.diff(curve)
        assert (diffs <= 1e-5).all(), f"Monotonicity violation detected in candidate {col}"


def test_dip_protocol_compliance():
    """Verifies that MultivariateSurvivalMLEngine satisfies the SurvivalPredictorProtocol contract."""
    engine = MultivariateSurvivalMLEngine()
    assert isinstance(engine, SurvivalPredictorProtocol), "Engine violates DIP SurvivalPredictorProtocol"


def test_semantic_layer_export_integrity(sample_ml_data, tmp_path):
    """Verifies export of dimensional marts with complete metadata and valid schema."""
    engine = MultivariateSurvivalMLEngine(penalizer=0.01)
    engine.fit(sample_ml_data)

    output_dir = str(tmp_path / "semantic_layer")
    paths = engine.export_semantic_layer(sample_ml_data, output_dir=output_dir)

    for key, path in paths.items():
        assert os.path.exists(path), f"Mart file missing: {path}"
        assert os.path.getsize(path) > 0, f"Empty file: {path}"

    df_hr = pd.read_parquet(paths["hazard_ratios_parquet"])
    assert "hazard_ratio" in df_hr.columns
    assert "p_value" in df_hr.columns
    assert "strategic_business_context" in df_hr.columns
    assert len(df_hr) > 0

    df_eval = pd.read_parquet(paths["evaluation_parquet"])
    assert "concordance_index_c" in df_eval.columns
    assert "integrated_brier_score" in df_eval.columns
    assert len(df_eval) > 0
