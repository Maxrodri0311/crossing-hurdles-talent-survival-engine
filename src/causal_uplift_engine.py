"""
src/causal_uplift_engine.py - Causal Uplift Survival & Knapsack Budget Optimizer
================================================================================
Implements:
1. Counterfactual Survival Estimation under Pearl's do-calculus framework:
   tau(x) = E[S(t=16 | do(A=1), x)] - E[S(t=16 | do(A=0), x)]
2. Heterogeneous Treatment Effect (HTE) 4-Quadrant Segmentation:
   - PERSUADABLES: High incremental lift (tau >= 0.15) -> Focus 100% of mentor capacity.
   - SURE_THINGS: Resilient graduates (s0 >= 0.70, tau < 0.15) -> Low ROI to intervene.
   - LOST_CAUSES: Structural disengagement (s1 <= 0.35, tau < 0.15) -> Alternative remediation.
   - MODERATE_RESPONDERS: Baseline responders (0.05 <= tau < 0.15).
3. Budget-Constrained Knapsack Optimization (Greedy Allocation):
   Maximizes aggregate student survival within a finite mentor hour budget B.
4. Financial ROI Modeling ($4,500 USD tuition preserved vs $135 USD mentor sprint cost).
5. Clean Architecture (DIP Protocol) and export of dimensional marts to Parquet & CSV.
"""

import os
import sys
from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Path resolution for standalone CLI execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.survival_ml import MultivariateSurvivalMLEngine
except ImportError:
    from survival_ml import MultivariateSurvivalMLEngine

# Windows UTF-8 output protection
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@runtime_checkable
class CausalUpliftProtocol(Protocol):
    """Abstract contract for causal uplift estimation and policy allocation (DIP)."""
    def estimate_individual_treatment_effects(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def segment_cohort(self, df_with_ite: pd.DataFrame) -> pd.DataFrame: ...
    def optimize_knapsack_budget(self, df_with_ite: pd.DataFrame, budget_hours: float) -> Dict[str, Any]: ...


class CausalInterventionSimulator:
    """
    Simulates counterfactual feature states under intervention do(A=1):
    - Express code review unblocking (tutor feedback latency <= 8.0h).
    - Homework remediation unblocking (assignment lag reduced by 2.5 days).
    - Architectural pairing (curriculum hurdle difficulty mitigated by 30%).
    """
    @classmethod
    def apply_counterfactual_treatment(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs counterfactual feature matrix under treatment A=1."""
        df_treated = df.copy()
        
        # Treatment Physics: Dedicated 1-on-1 sprint removes operational bottlenecks
        df_treated["tutor_feedback_latency_hours"] = np.minimum(
            df_treated["tutor_feedback_latency_hours"].astype(float),
            8.0
        )
        df_treated["assignment_lag_days"] = np.maximum(
            0.0,
            df_treated["assignment_lag_days"].astype(float) - 2.5
        )
        df_treated["hurdle_difficulty_index"] = (
            df_treated["hurdle_difficulty_index"].astype(float) * 0.70
        )
        return df_treated


class CausalUpliftEngine:
    """
    Production Causal Uplift & Intervention Allocation Engine.
    """
    TUITION_VALUE_USD = 4500.0          # Tuition revenue preserved per graduate
    MENTOR_HOURLY_RATE_USD = 45.0       # Cost per mentor hour
    HOURS_PER_INTERVENTION = 3.0        # Duration of dedicated 1-on-1 sprint
    COST_PER_INTERVENTION = 135.0       # 3.0h * $45/h = $135

    def __init__(self, ml_engine: Optional[MultivariateSurvivalMLEngine] = None, penalizer: float = 0.01):
        self.ml_engine = ml_engine or MultivariateSurvivalMLEngine(penalizer=penalizer)
        self.simulator = CausalInterventionSimulator()
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CausalUpliftEngine":
        """Fits the underlying survival model if not already fitted."""
        if not self.ml_engine._is_fitted:
            self.ml_engine.fit(df)
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Engine has not been fitted. Call fit() before inference.")

    def estimate_individual_treatment_effects(self, df: pd.DataFrame, target_week: float = 16.0) -> pd.DataFrame:
        """
        Estimates S0(t=16 | x), S1(t=16 | x), and ITE tau_i = S1 - S0 for every candidate.
        """
        self._check_fitted()
        
        # 1. Baseline factual survival probability at graduation milestone
        s0_baseline = self.ml_engine.predict_survival_at_horizon(df, time=target_week)
        s0_baseline = np.clip(s0_baseline, 0.0, 1.0)

        # 2. Counterfactual treated survival probability at graduation milestone
        df_treated = self.simulator.apply_counterfactual_treatment(df)
        s1_treated = self.ml_engine.predict_survival_at_horizon(df_treated, time=target_week)
        s1_treated = np.clip(s1_treated, 0.0, 1.0)

        # 3. Individual Treatment Effect (ITE)
        ite_tau = np.round(s1_treated - s0_baseline, 4)

        results = df.copy()
        results["s0_baseline_survival"] = np.round(s0_baseline, 4)
        results["s1_treated_survival"] = np.round(s1_treated, 4)
        results["ite_survival_uplift"] = ite_tau
        
        # Expected Net Monetary Value of treating this specific candidate
        expected_gross_gain = ite_tau * self.TUITION_VALUE_USD
        results["expected_net_value_usd"] = np.round(expected_gross_gain - self.COST_PER_INTERVENTION, 2)

        return results

    def segment_cohort(self, df_with_ite: pd.DataFrame) -> pd.DataFrame:
        """
        Partitions candidates into the 4 Uplift Quadrants:
        - PERSUADABLE: ite_tau >= 0.15 (High impact responders)
        - SURE_THING: s0 >= 0.70 and ite_tau < 0.15 (Will graduate anyway)
        - LOST_CAUSE: s1 <= 0.35 and ite_tau < 0.15 (Disengaged, needs fundamental reset)
        - MODERATE_RESPONDER: 0.05 <= ite_tau < 0.15 (Positive but moderate lift)
        """
        df = df_with_ite.copy()
        s0 = df["s0_baseline_survival"]
        s1 = df["s1_treated_survival"]
        tau = df["ite_survival_uplift"]

        conditions = [
            tau >= 0.15,
            (s0 >= 0.70) & (tau < 0.15),
            (s1 <= 0.35) & (tau < 0.15),
            (tau >= 0.05) & (tau < 0.15)
        ]
        choices = [
            "PERSUADABLE",
            "SURE_THING",
            "LOST_CAUSE",
            "MODERATE_RESPONDER"
        ]
        df["uplift_quadrant"] = np.select(conditions, choices, default="NEUTRAL_RESPONDER")
        return df

    def optimize_knapsack_budget(
        self,
        df_segmented: pd.DataFrame,
        budget_hours: float
    ) -> Dict[str, Any]:
        """
        Solves greedy knapsack resource allocation:
        max sum(pi_i * tau_i * V_tuition) s.t. sum(pi_i * 3.0h) <= budget_hours.
        """
        max_interventions = int(budget_hours // self.HOURS_PER_INTERVENTION)
        df_sorted = df_segmented.sort_values(by="ite_survival_uplift", ascending=False).reset_index(drop=True)

        selected_slice = df_sorted.head(max_interventions)
        n_selected = len(selected_slice)

        incremental_graduates_rescued = float(selected_slice["ite_survival_uplift"].sum())
        total_investment_cost_usd = n_selected * self.COST_PER_INTERVENTION
        rescued_tuition_revenue_usd = incremental_graduates_rescued * self.TUITION_VALUE_USD
        net_financial_profit_usd = rescued_tuition_revenue_usd - total_investment_cost_usd
        
        roi_pct = 0.0
        if total_investment_cost_usd > 0:
            roi_pct = (net_financial_profit_usd / total_investment_cost_usd) * 100.0

        return {
            "budget_hours": budget_hours,
            "max_interventions": max_interventions,
            "candidates_treated": n_selected,
            "incremental_graduates_rescued": round(incremental_graduates_rescued, 1),
            "total_investment_cost_usd": round(total_investment_cost_usd, 2),
            "rescued_tuition_revenue_usd": round(rescued_tuition_revenue_usd, 2),
            "net_financial_profit_usd": round(net_financial_profit_usd, 2),
            "net_roi_percent": round(roi_pct, 1),
            "avg_uplift_in_treated": round(float(selected_slice["ite_survival_uplift"].mean()), 4) if n_selected > 0 else 0.0
        }

    def evaluate_budget_scenarios(
        self,
        df_segmented: pd.DataFrame,
        scenarios_hours: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Evaluates Knapsack optimization across a spectrum of operational budgets."""
        if scenarios_hours is None:
            scenarios_hours = [600.0, 1500.0, 3000.0, 6000.0, 12000.0]

        records = []
        for b_hours in scenarios_hours:
            res = self.optimize_knapsack_budget(df_segmented, budget_hours=b_hours)
            records.append(res)
        return pd.DataFrame(records)

    def export_semantic_layer(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/semantic_layer",
        target_budget_hours: float = 3000.0
    ) -> Dict[str, str]:
        """Exports causal uplift segments, budget allocation scenarios, and student prescriptions."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        df_ite = self.estimate_individual_treatment_effects(df)
        df_segmented = self.segment_cohort(df_ite)

        # 1. Export Causal Uplift Quadrants Dimension Mart
        quadrant_summary = df_segmented.groupby("uplift_quadrant").agg(
            candidate_count=("candidate_id", "count"),
            avg_baseline_survival=("s0_baseline_survival", "mean"),
            avg_treated_survival=("s1_treated_survival", "mean"),
            avg_ite_uplift=("ite_survival_uplift", "mean"),
            avg_net_value_usd=("expected_net_value_usd", "mean")
        ).reset_index()

        policy_map = {
            "PERSUADABLE": "Priority 1: Allocate 100% of available 1-on-1 mentor sprint capacity",
            "MODERATE_RESPONDER": "Priority 2: Group asynchronous workshops & office hours",
            "SURE_THING": "Autonomous track: Zero extra mentor spend needed, ensure unblocking SLA",
            "LOST_CAUSE": "Remediation track: 2-week curriculum leveling or cohort deferral",
            "NEUTRAL_RESPONDER": "Standard academic monitoring"
        }
        quadrant_summary["prescribed_strategic_policy"] = quadrant_summary["uplift_quadrant"].map(
            lambda x: policy_map.get(x, "Standard monitoring")
        )

        p_quad_parquet = os.path.join(output_dir, "dim_causal_uplift_segments.parquet")
        p_quad_csv = os.path.join(output_dir, "dim_causal_uplift_segments.csv")
        quadrant_summary.to_parquet(p_quad_parquet, index=False)
        quadrant_summary.to_csv(p_quad_csv, index=False)
        paths["uplift_segments_parquet"] = p_quad_parquet
        paths["uplift_segments_csv"] = p_quad_csv

        # 2. Export Knapsack Budget Allocation Scenarios Mart
        df_scenarios = self.evaluate_budget_scenarios(df_segmented)
        p_scen_parquet = os.path.join(output_dir, "dim_knapsack_budget_allocations.parquet")
        p_scen_csv = os.path.join(output_dir, "dim_knapsack_budget_allocations.csv")
        df_scenarios.to_parquet(p_scen_parquet, index=False)
        df_scenarios.to_csv(p_scen_csv, index=False)
        paths["knapsack_scenarios_parquet"] = p_scen_parquet
        paths["knapsack_scenarios_csv"] = p_scen_csv

        # 3. Export Fact Student Causal Prescriptions (Tagging with 3,000h budget selection)
        max_interventions = int(target_budget_hours // self.HOURS_PER_INTERVENTION)
        df_sorted = df_segmented.sort_values(by="ite_survival_uplift", ascending=False).copy()
        
        selection_flags = ["SELECTED_IN_TARGET_BUDGET"] * min(max_interventions, len(df_sorted))
        if len(df_sorted) > max_interventions:
            selection_flags += ["UNALLOCATED_BUDGET_EXHAUSTED"] * (len(df_sorted) - max_interventions)
        df_sorted["knapsack_allocation_status"] = selection_flags

        export_cols = [
            "candidate_id", "cohort_id", "track", "prior_experience",
            "s0_baseline_survival", "s1_treated_survival", "ite_survival_uplift",
            "expected_net_value_usd", "uplift_quadrant", "knapsack_allocation_status"
        ]
        available_cols = [c for c in export_cols if c in df_sorted.columns]
        fact_prescriptions = df_sorted[available_cols]

        p_fact_parquet = os.path.join(output_dir, "fact_student_causal_prescriptions.parquet")
        p_fact_csv = os.path.join(output_dir, "fact_student_causal_prescriptions.csv")
        fact_prescriptions.to_parquet(p_fact_parquet, index=False)
        fact_prescriptions.to_csv(p_fact_csv, index=False)
        paths["causal_prescriptions_parquet"] = p_fact_parquet
        paths["causal_prescriptions_csv"] = p_fact_csv

        return paths


def run_causal_uplift_pipeline(
    data_path: str = "data/raw_dataset.parquet",
    output_dir: str = "data/semantic_layer"
) -> Tuple[CausalUpliftEngine, Dict[str, Any]]:
    """Convenience pipeline to execute causal uplift modeling and knapsack optimization."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at {data_path}")

    df_raw = pd.read_parquet(data_path)
    engine = CausalUpliftEngine()
    engine.fit(df_raw)

    df_ite = engine.estimate_individual_treatment_effects(df_raw)
    df_segmented = engine.segment_cohort(df_ite)
    df_scenarios = engine.evaluate_budget_scenarios(df_segmented)
    exported_paths = engine.export_semantic_layer(df_raw, output_dir=output_dir)

    summary = {
        "quadrant_counts": df_segmented["uplift_quadrant"].value_counts().to_dict(),
        "scenarios": df_scenarios.to_dict(orient="records"),
        "exported_paths": exported_paths,
        "total_records": len(df_raw)
    }
    return engine, summary


if __name__ == "__main__":
    print("=" * 80)
    print("⚡ RUNNING CAUSAL UPLIFT ENGINE & KNAPSACK BUDGET OPTIMIZER (FASE C)")
    print("=" * 80)

    engine, summary = run_causal_uplift_pipeline()
    print(f"\n[+] Candidates Evaluated: {summary['total_records']:,}")
    print("\n[+] Causal Uplift 4-Quadrant Partitioning:")
    print("-" * 80)
    for quad, count in summary["quadrant_counts"].items():
        pct = (count / summary["total_records"]) * 100.0
        print(f"    * {quad:<20}: {count:>6,} candidates ({pct:>5.1f}%)")
    print("-" * 80)

    print("\n[+] Knapsack Budget Allocation Scenarios (Tuition: $4,500 | Sprint Cost: $135):")
    print("-" * 80)
    print(f"{'Budget (Hours)':<14} | {'Treated':<8} | {'Rescued Grad':<13} | {'Cost (USD)':<12} | {'Revenue (USD)':<14} | {'Net ROI':<8}")
    print("-" * 80)
    for sc in summary["scenarios"]:
        b_str = f"{sc['budget_hours']:,.0f} hrs"
        tr_str = f"{sc['candidates_treated']:,}"
        rg_str = f"{sc['incremental_graduates_rescued']:,.1f}"
        cost_str = f"${sc['total_investment_cost_usd']:,.0f}"
        rev_str = f"${sc['rescued_tuition_revenue_usd']:,.0f}"
        roi_str = f"{sc['net_roi_percent']:,.1f}%"
        print(f"{b_str:<14} | {tr_str:<8} | {rg_str:<13} | {cost_str:<12} | {rev_str:<14} | {roi_str:<8}")
    print("=" * 80)
