"""
src/survival_ml.py - Advanced Survival Machine Learning & Calibration Engine
=============================================================================
Implements:
1. Feature Engineering with high-order interaction terms (non-STEM background x mentor latency x hurdle difficulty).
2. Regularized Multivariate Cox Proportional Hazards Model (lifelines).
3. Harrell's Concordance Index (C-Index) discrimination metric.
4. Time-Dependent Brier Score and Integrated Brier Score (IBS) via Inverse Probability
   of Censoring Weighting (IPCW).
5. Clean Architecture (Dependency Inversion Principle) with abstract protocols.
6. Export of dimensional marts to Apache Parquet and CSV for C-level dashboards.
"""

import os
import sys
from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

# Windows UTF-8 output protection
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@runtime_checkable
class SurvivalPredictorProtocol(Protocol):
    """Abstract contract for survival machine learning estimators (DIP)."""
    def fit(self, df: pd.DataFrame) -> "SurvivalPredictorProtocol": ...
    def predict_partial_hazard(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_survival_at_horizon(self, df: pd.DataFrame, time: float) -> np.ndarray: ...
    def compute_concordance_index(self, df: pd.DataFrame) -> float: ...


class SurvivalFeatureTransformer:
    """
    Transforms raw EdTech candidate telemetry into high-order interaction matrices.
    Captures non-linear dynamics between candidate background and operational bottlenecks.
    """
    FEATURE_COLS = [
        "weekly_hours_dedicated",
        "assignment_lag_days",
        "tutor_feedback_latency_hours",
        "hurdle_difficulty_index",
        "assessment_score_avg",
        "is_career_switcher",
        "is_stem_graduate",
        "interaction_switcher_latency",
        "interaction_switcher_difficulty",
        "interaction_hours_deficit_lag"
    ]

    @classmethod
    def transform(cls, df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
        """Transforms raw input DataFrame into engineered feature matrix."""
        df_feat = pd.DataFrame(index=df.index)

        # Baseline numerical covariates
        df_feat["weekly_hours_dedicated"] = df["weekly_hours_dedicated"].astype(float)
        df_feat["assignment_lag_days"] = df["assignment_lag_days"].astype(float)
        df_feat["tutor_feedback_latency_hours"] = df["tutor_feedback_latency_hours"].astype(float)
        df_feat["hurdle_difficulty_index"] = df["hurdle_difficulty_index"].astype(float)
        df_feat["assessment_score_avg"] = df["assessment_score_avg"].astype(float)

        # Categorical indicators for prior experience
        exp = df["prior_experience"].astype(str)
        df_feat["is_career_switcher"] = (exp == "Non-Technical Career Switcher").astype(float)
        df_feat["is_stem_graduate"] = (exp == "STEM University Graduate").astype(float)

        # High-order non-linear interactions (The 'Hurdle' Physics)
        # 1. Career switchers suffering from slow mentor feedback
        df_feat["interaction_switcher_latency"] = (
            df_feat["is_career_switcher"] * (df_feat["tutor_feedback_latency_hours"] / 24.0)
        )
        # 2. Career switchers hitting complex architectural hurdles
        df_feat["interaction_switcher_difficulty"] = (
            df_feat["is_career_switcher"] * df_feat["hurdle_difficulty_index"]
        )
        # 3. Weekly study hour deficit compounded by assignment lag
        hours_deficit = np.maximum(0.0, 20.0 - df_feat["weekly_hours_dedicated"])
        df_feat["interaction_hours_deficit_lag"] = hours_deficit * df_feat["assignment_lag_days"]

        # Targets for model training
        if include_targets and "duration_weeks" in df.columns and "event_observed" in df.columns:
            df_feat["duration_weeks"] = df["duration_weeks"].astype(float)
            df_feat["event_observed"] = df["event_observed"].astype(int)

        return df_feat


class MultivariateSurvivalMLEngine:
    """
    Enterprise-grade Survival Machine Learning Engine with IPCW Brier Score calibration,
    Harrell's C-index evaluation, and dimensional mart synthesis.
    """
    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fitter: Optional[CoxPHFitter] = None
        self.censoring_km: Optional[KaplanMeierFitter] = None
        self.feature_transformer = SurvivalFeatureTransformer()
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MultivariateSurvivalMLEngine":
        """Fits the regularized Cox proportional hazards model and censoring estimator."""
        df_train = self.feature_transformer.transform(df, include_targets=True)

        self.fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.fitter.fit(
            df_train,
            duration_col="duration_weeks",
            event_col="event_observed",
            show_progress=False
        )

        # Fit Kaplan-Meier on censoring distribution for IPCW weighting
        # (reverse event indicator: 1 - delta)
        self.censoring_km = KaplanMeierFitter()
        censoring_indicator = 1 - df_train["event_observed"]
        self.censoring_km.fit(
            durations=df_train["duration_weeks"],
            event_observed=censoring_indicator
        )

        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted or self.fitter is None:
            raise RuntimeError("Model has not been fitted. Call fit() before inference.")

    def predict_partial_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """Returns relative partial hazard score: exp(beta * x)."""
        self._check_fitted()
        df_feat = self.feature_transformer.transform(df, include_targets=False)
        return self.fitter.predict_partial_hazard(df_feat).to_numpy()

    def predict_survival_at_horizon(self, df: pd.DataFrame, time: float) -> np.ndarray:
        """Returns S(time | x) for each individual at target week."""
        self._check_fitted()
        df_feat = self.feature_transformer.transform(df, include_targets=False)
        surv_df = self.fitter.predict_survival_function(df_feat, times=[time])
        return surv_df.iloc[0].to_numpy()

    def predict_survival_matrix(self, df: pd.DataFrame, times: List[float]) -> pd.DataFrame:
        """Returns full survival probability matrix for a sequence of time points."""
        self._check_fitted()
        df_feat = self.feature_transformer.transform(df, include_targets=False)
        return self.fitter.predict_survival_function(df_feat, times=times)

    def compute_concordance_index(self, df: Optional[pd.DataFrame] = None) -> float:
        """Computes Harrell's Concordance Index on evaluation dataset or training data."""
        self._check_fitted()
        if df is None:
            return float(self.fitter.concordance_index_)
        df_eval = self.feature_transformer.transform(df, include_targets=True)
        from lifelines.utils import concordance_index
        partial_hazard = self.fitter.predict_partial_hazard(df_eval).to_numpy()
        # Higher hazard -> lower survival time: concordance compares (-hazard) vs duration
        return float(concordance_index(df_eval["duration_weeks"], -partial_hazard, df_eval["event_observed"]))

    def compute_time_dependent_brier_score(self, df: pd.DataFrame, time: float) -> float:
        """
        Computes the Graf et al. (1999) Brier Score at horizon t using Inverse Probability
        of Censoring Weighting (IPCW).
        BS(t) = (1/N) * sum_i [ (S_hat(t|x_i))^2 * I(T_i <= t, d_i=1) / G(T_i) +
                                (1 - S_hat(t|x_i))^2 * I(T_i > t) / G(t) ]
        """
        self._check_fitted()
        n = len(df)
        if n == 0:
            return 0.0

        durations = df["duration_weeks"].to_numpy(dtype=float)
        events = df["event_observed"].to_numpy(dtype=int)
        surv_preds = self.predict_survival_at_horizon(df, time)

        # Evaluate G(t) (censoring survival probability at time t)
        g_t = float(self.censoring_km.survival_function_at_times(time).iloc[0])
        g_t = max(g_t, 1e-5)  # avoid division by zero

        # Evaluate G(T_i) for subjects with event prior to t
        g_ti = self.censoring_km.survival_function_at_times(durations).to_numpy()
        g_ti = np.maximum(g_ti, 1e-5)

        # Term 1: Events occurred prior to or at time t
        event_mask = (durations <= time) & (events == 1)
        term1 = np.sum((surv_preds[event_mask] ** 2) / g_ti[event_mask])

        # Term 2: Individuals still at risk at time t
        surv_mask = (durations > time)
        term2 = np.sum(((1.0 - surv_preds[surv_mask]) ** 2) / g_t)

        return float((term1 + term2) / n)

    def compute_integrated_brier_score(
        self,
        df: pd.DataFrame,
        eval_times: Optional[List[float]] = None
    ) -> Tuple[float, Dict[float, float]]:
        """
        Computes the Integrated Brier Score (IBS) across time horizons [1, 16] weeks.
        Returns: (IBS_value, dictionary_of_time_brier_scores).
        """
        if eval_times is None:
            eval_times = [float(t) for t in range(2, 16, 2)]

        brier_by_time = {}
        for t in eval_times:
            brier_by_time[t] = self.compute_time_dependent_brier_score(df, t)

        times_arr = np.array(eval_times)
        brier_arr = np.array([brier_by_time[t] for t in eval_times])

        # Exact trapezoidal numerical integration (NumPy 2.x robust)
        integral = float(0.5 * np.sum((brier_arr[1:] + brier_arr[:-1]) * (times_arr[1:] - times_arr[:-1])))
        ibs = float(integral / (times_arr[-1] - times_arr[0]))
        return ibs, brier_by_time

    def get_hazard_ratios_report(self) -> pd.DataFrame:
        """
        Extracts structured Hazard Ratios (exp(coef)), 95% Confidence Intervals,
        standard errors, and z-test p-values.
        """
        self._check_fitted()
        summary = self.fitter.summary.copy()
        summary = summary.reset_index().rename(columns={"covariate": "feature_name"})

        # Human-readable strategic business interpretation
        interpretations = {
            "weekly_hours_dedicated": "Decrease in hazard per additional weekly study hour",
            "assignment_lag_days": "Hazard expansion per day of homework submission delay",
            "tutor_feedback_latency_hours": "Hazard escalation per hour of tutor feedback delay",
            "hurdle_difficulty_index": "Curriculum architectural complexity hazard multiplier",
            "assessment_score_avg": "Protective hazard reduction per point of technical score",
            "is_career_switcher": "Baseline vulnerability for candidates transitioning from non-tech",
            "is_stem_graduate": "Protective buffer from prior formal engineering training",
            "interaction_switcher_latency": "Compound vulnerability: Career Switcher exposed to slow feedback",
            "interaction_switcher_difficulty": "Compound vulnerability: Career Switcher hitting complex modules",
            "interaction_hours_deficit_lag": "Cascading risk: Hour deficit (<20h) compounded by submission lag"
        }

        summary["strategic_business_context"] = summary["feature_name"].map(
            lambda x: interpretations.get(x, "High-order analytical interaction term")
        )

        # Reorder and format columns
        columns_map = {
            "feature_name": "feature_name",
            "coef": "log_hazard_coef",
            "exp(coef)": "hazard_ratio",
            "se(coef)": "standard_error",
            "exp(coef) lower 95%": "hr_ci_95_lower",
            "exp(coef) upper 95%": "hr_ci_95_upper",
            "p": "p_value",
            "strategic_business_context": "strategic_business_context"
        }
        
        available_cols = [c for c in columns_map if c in summary.columns]
        summary = summary[available_cols].rename(columns=columns_map)
        return summary

    def export_semantic_layer(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/semantic_layer"
    ) -> Dict[str, str]:
        """
        Exports model coefficients and formal calibration metrics to Apache Parquet & CSV
        for Power BI and Tableau ingestion.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # 1. Export Coefficients & Hazard Ratios Mart
        df_hr = self.get_hazard_ratios_report()
        p_hr_parquet = os.path.join(output_dir, "dim_multivariate_hazard_ratios.parquet")
        p_hr_csv = os.path.join(output_dir, "dim_multivariate_hazard_ratios.csv")
        df_hr.to_parquet(p_hr_parquet, index=False)
        df_hr.to_csv(p_hr_csv, index=False)
        paths["hazard_ratios_parquet"] = p_hr_parquet
        paths["hazard_ratios_csv"] = p_hr_csv

        # 2. Export Model Evaluation & Calibration Mart
        c_index = self.compute_concordance_index(df)
        ibs, brier_curve = self.compute_integrated_brier_score(df)

        eval_records = []
        for week_horizon, bs in brier_curve.items():
            eval_records.append({
                "model_architecture": "Regularized Multivariate Cox (ElasticNet L2)",
                "concordance_index_c": round(c_index, 4),
                "integrated_brier_score": round(ibs, 4),
                "evaluation_horizon_week": week_horizon,
                "time_dependent_brier_score": round(bs, 4),
                "penalizer": self.penalizer,
                "sample_size": len(df)
            })

        df_eval = pd.DataFrame(eval_records)
        p_eval_parquet = os.path.join(output_dir, "dim_survival_model_evaluation.parquet")
        p_eval_csv = os.path.join(output_dir, "dim_survival_model_evaluation.csv")
        df_eval.to_parquet(p_eval_parquet, index=False)
        df_eval.to_csv(p_eval_csv, index=False)
        paths["evaluation_parquet"] = p_eval_parquet
        paths["evaluation_csv"] = p_eval_csv

        return paths


def train_and_export_survival_ml(
    data_path: str = "data/raw_dataset.parquet",
    output_dir: str = "data/semantic_layer",
    penalizer: float = 0.01
) -> Tuple[MultivariateSurvivalMLEngine, Dict[str, Any]]:
    """Convenience pipeline function to train, evaluate, and export survival ML marts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset missing at {data_path}")

    df_raw = pd.read_parquet(data_path)
    engine = MultivariateSurvivalMLEngine(penalizer=penalizer)
    engine.fit(df_raw)

    c_index = engine.compute_concordance_index(df_raw)
    ibs, brier_curve = engine.compute_integrated_brier_score(df_raw)
    exported_paths = engine.export_semantic_layer(df_raw, output_dir=output_dir)

    metrics = {
        "concordance_index": c_index,
        "integrated_brier_score": ibs,
        "brier_curve": brier_curve,
        "exported_paths": exported_paths,
        "total_records": len(df_raw)
    }
    return engine, metrics


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 TRAINING ADVANCED SURVIVAL MACHINE LEARNING ENGINE (FASE A)")
    print("=" * 70)
    
    engine, summary = train_and_export_survival_ml()
    print(f"[+] Records Evaluated:         {summary['total_records']:,}")
    print(f"[+] Harrell's C-Index:         {summary['concordance_index']:.4f} (Benchmark: >0.70)")
    print(f"[+] Integrated Brier Score:    {summary['integrated_brier_score']:.4f} (Benchmark: <0.20)")
    print("\n[+] Brier Score Calibration Curve by Week Horizon:")
    for w, bs in summary["brier_curve"].items():
        print(f"    - Week {w:2.0f}: Brier Score = {bs:.4f}")

    print("\n[+] Top Hazard Ratio Multipliers (Model Findings):")
    df_hr = engine.get_hazard_ratios_report()
    top_hr = df_hr.sort_values(by="hazard_ratio", ascending=False).head(5)
    for _, row in top_hr.iterrows():
        print(f"    * {row['feature_name']:<32} | HR: {row['hazard_ratio']:.2f}x (95% CI: [{row['hr_ci_95_lower']:.2f}, {row['hr_ci_95_upper']:.2f}]) | p={row['p_value']:.4e}")
    print("=" * 70)
