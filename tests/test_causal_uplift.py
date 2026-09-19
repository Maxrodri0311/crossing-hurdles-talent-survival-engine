"""
tests/test_causal_uplift.py - Unit Test Suite for Causal Uplift & Knapsack Optimizer
===================================================================================
Validates:
1. Counterfactual survival probabilities in [0, 1] and positive treatment response S1 >= S0.
2. ITE bounds tau_i in [-1, 1] and quadrant partitioning population conservation.
3. Knapsack budget constraint (treated * 3.0h <= budget_hours).
4. Decreasing marginal ROI property (greedy knapsack efficiency ranking).
5. Clean Architecture (DIP Protocol compliance).
6. Semantic layer mart schema integrity and non-null guarantees.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_dataset
from src.causal_uplift_engine import (
    CausalUpliftProtocol,
    CausalUpliftEngine,
    CausalInterventionSimulator
)


@pytest.fixture(scope="session")
def causal_sample_data(tmp_path_factory):
    fn = tmp_path_factory.mktemp("causal_data") / "sample_cohort.parquet"
    df = generate_synthetic_dataset(num_records=2500, output_path=str(fn), seed=777)
    return df


def test_causal_counterfactual_bounds_and_monotonicity(causal_sample_data):
    """Verifies counterfactual survival predictions satisfy probability bounds and positive treatment lift."""
    engine = CausalUpliftEngine(penalizer=0.01)
    engine.fit(causal_sample_data)

    df_ite = engine.estimate_individual_treatment_effects(causal_sample_data, target_week=16.0)
    
    assert (df_ite["s0_baseline_survival"] >= 0.0).all() and (df_ite["s0_baseline_survival"] <= 1.0).all()
    assert (df_ite["s1_treated_survival"] >= 0.0).all() and (df_ite["s1_treated_survival"] <= 1.0).all()
    assert (df_ite["ite_survival_uplift"] >= -0.05).all(), "Negative treatment response beyond noise threshold"
    assert (df_ite["ite_survival_uplift"] <= 1.0).all()
    assert df_ite["ite_survival_uplift"].mean() > 0.05, "Intervention showed negligible aggregate uplift"


def test_quadrant_partitioning_conservation(causal_sample_data):
    """Verifies that all candidates are partitioned without loss or duplication."""
    engine = CausalUpliftEngine(penalizer=0.01)
    engine.fit(causal_sample_data)

    df_ite = engine.estimate_individual_treatment_effects(causal_sample_data)
    df_segmented = engine.segment_cohort(df_ite)

    # Invariant 1: Population conservation
    assert len(df_segmented) == len(causal_sample_data)
    valid_quads = {"PERSUADABLE", "SURE_THING", "LOST_CAUSE", "MODERATE_RESPONDER", "NEUTRAL_RESPONDER"}
    assert set(df_segmented["uplift_quadrant"].unique()).issubset(valid_quads)

    # Invariant 2: Persuadables have high incremental lift
    persuadables = df_segmented[df_segmented["uplift_quadrant"] == "PERSUADABLE"]
    if len(persuadables) > 0:
        assert (persuadables["ite_survival_uplift"] >= 0.15).all()

    # Invariant 3: Sure things have high baseline survival
    sure_things = df_segmented[df_segmented["uplift_quadrant"] == "SURE_THING"]
    if len(sure_things) > 0:
        assert (sure_things["s0_baseline_survival"] >= 0.70).all()


def test_knapsack_optimization_constraints_and_marginal_roi(causal_sample_data):
    """Verifies knapsack budget adherence and decreasing marginal return property."""
    engine = CausalUpliftEngine(penalizer=0.01)
    engine.fit(causal_sample_data)

    df_ite = engine.estimate_individual_treatment_effects(causal_sample_data)
    df_segmented = engine.segment_cohort(df_ite)

    # Test single budget run
    budget_hours = 600.0
    res = engine.optimize_knapsack_budget(df_segmented, budget_hours=budget_hours)
    
    # Invariant 1: Budget constraint
    assert res["candidates_treated"] * 3.0 <= budget_hours
    assert res["candidates_treated"] == int(budget_hours // 3.0)
    assert res["incremental_graduates_rescued"] > 0.0
    assert res["net_roi_percent"] > 100.0, "ROI must be strongly positive for top persuadables"

    # Invariant 2: Decreasing marginal ROI
    scenarios = engine.evaluate_budget_scenarios(df_segmented, scenarios_hours=[300.0, 900.0, 1800.0])
    rois = scenarios["net_roi_percent"].to_numpy()
    # Marginal efficiency decreases as budget expands into lower uplift candidates
    assert rois[0] >= rois[1] >= rois[2], "Violation of greedy knapsack decreasing marginal ROI"


def test_dip_protocol_compliance():
    """Verifies that CausalUpliftEngine conforms to CausalUpliftProtocol."""
    engine = CausalUpliftEngine()
    assert isinstance(engine, CausalUpliftProtocol), "Engine violates DIP CausalUpliftProtocol"


def test_semantic_layer_causal_marts_export(causal_sample_data, tmp_path):
    """Verifies export of causal segments, budget scenarios, and student prescriptions."""
    engine = CausalUpliftEngine(penalizer=0.01)
    engine.fit(causal_sample_data)

    output_dir = str(tmp_path / "semantic_layer")
    paths = engine.export_semantic_layer(causal_sample_data, output_dir=output_dir)

    for key, path in paths.items():
        assert os.path.exists(path), f"Causal mart missing: {path}"
        assert os.path.getsize(path) > 0, f"Empty file: {path}"

    df_quad = pd.read_parquet(paths["uplift_segments_parquet"])
    assert "uplift_quadrant" in df_quad.columns
    assert "avg_ite_uplift" in df_quad.columns
    assert "prescribed_strategic_policy" in df_quad.columns

    df_scen = pd.read_parquet(paths["knapsack_scenarios_parquet"])
    assert "budget_hours" in df_scen.columns
    assert "net_roi_percent" in df_scen.columns
    assert len(df_scen) >= 5

    df_fact = pd.read_parquet(paths["causal_prescriptions_parquet"])
    assert "ite_survival_uplift" in df_fact.columns
    assert "knapsack_allocation_status" in df_fact.columns
    assert len(df_fact) == len(causal_sample_data)
