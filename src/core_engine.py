"""
src/core_engine.py - Core Survival Analytics & Explainable AI Engine
====================================================================
Implements Kaplan-Meier Product-Limit Estimator, Actuarial Life Tables,
and Explainable Cox Proportional Hazards Ratio Modeling with DuckDB
in-memory OLAP under strict Clean Architecture (Dependency Inversion Principle).
"""

import os
import sys
from typing import Protocol, Optional, Dict, Any, List
import duckdb
import numpy as np
import pandas as pd

# Windows UTF-8 physics
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class AnalyticalStorageProtocol(Protocol):
    """Abstract contract for analytical persistence and execution (DIP)."""
    def execute_query(self, query: str) -> pd.DataFrame: ...


class DuckDBStorageAdapter:
    """Concrete infrastructure adapter for DuckDB in-memory OLAP execution."""
    def __init__(self, connection: Optional[duckdb.DuckDBPyConnection] = None):
        self.conn = connection or duckdb.connect(":memory:")

    def execute_query(self, query: str) -> pd.DataFrame:
        return self.conn.execute(query).df()


class AnalyticsEngine:
    """
    Domain Analytics Engine decoupled from concrete storage/IO dependencies.
    Computes rigorous non-parametric survival metrics and explainable hazard ratios.
    """
    def __init__(
        self,
        storage: AnalyticalStorageProtocol,
        data_path: str = "data/raw_dataset.parquet"
    ):
        self.storage = storage
        self.data_path = data_path.replace("\\", "/")

    def _ensure_data_exists(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

    def compute_kaplan_meier_overall(self) -> pd.DataFrame:
        """
        Computes the Kaplan-Meier Product-Limit Estimator with Greenwood's standard error:
        S(t) = Prod_{t_i <= t} (1 - d_i / n_i)
        SE(S(t)) = S(t) * sqrt( sum( d_i / (n_i * (n_i - d_i)) ) )
        """
        self._ensure_data_exists()
        
        # Extract distinct event times (binned into 0.5-week steps for clean curves)
        query = f"""
            WITH timeline AS (
                SELECT 
                    ROUND(duration_weeks * 2) / 2.0 AS time_week,
                    SUM(event_observed) AS events_count,
                    COUNT(*) AS total_records
                FROM read_parquet('{self.data_path}')
                GROUP BY 1
                ORDER BY 1 ASC
            )
            SELECT 
                time_week,
                events_count,
                (total_records - events_count) AS censored_count,
                total_records
            FROM timeline;
        """
        df_events = self.storage.execute_query(query)
        if df_events.empty:
            return pd.DataFrame()

        # Compute total population
        query_total = f"SELECT COUNT(*) as total_pop FROM read_parquet('{self.data_path}')"
        total_pop = int(self.storage.execute_query(query_total).iloc[0]["total_pop"])

        km_records = []
        n_at_risk = total_pop
        survival_prob = 1.0
        greenwood_sum = 0.0

        for _, row in df_events.iterrows():
            t = float(row["time_week"])
            d = int(row["events_count"])
            c = int(row["censored_count"])

            if n_at_risk <= 0:
                break

            # Instantaneous hazard at time t
            hazard = (d / n_at_risk) if n_at_risk > 0 else 0.0
            survival_prob *= (1.0 - hazard)

            # Greenwood variance component
            if n_at_risk > d and n_at_risk > 0 and d > 0:
                greenwood_sum += d / (n_at_risk * (n_at_risk - d))

            std_err = survival_prob * np.sqrt(greenwood_sum)
            ci_lower = max(0.0, survival_prob - 1.96 * std_err)
            ci_upper = min(1.0, survival_prob + 1.96 * std_err)
            cum_hazard = -np.log(survival_prob) if survival_prob > 0 else np.nan

            km_records.append({
                "time_week": t,
                "n_at_risk": n_at_risk,
                "events_dropouts": d,
                "censored": c,
                "survival_probability": round(survival_prob, 4),
                "cumulative_hazard": round(cum_hazard, 4) if not np.isnan(cum_hazard) else 9.999,
                "std_error": round(std_err, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4)
            })

            n_at_risk -= (d + c)

        return pd.DataFrame(km_records)

    def compute_stratified_kaplan_meier(self, strata_col: str = "prior_experience") -> pd.DataFrame:
        """
        Computes stratified Kaplan-Meier curves across cohorts or student backgrounds.
        """
        self._ensure_data_exists()
        query_strata = f"SELECT DISTINCT {strata_col} FROM read_parquet('{self.data_path}') ORDER BY 1"
        strata_values = self.storage.execute_query(query_strata)[strata_col].tolist()

        all_curves = []
        for s_val in strata_values:
            query = f"""
                WITH timeline AS (
                    SELECT 
                        ROUND(duration_weeks * 2) / 2.0 AS time_week,
                        SUM(event_observed) AS events_count,
                        COUNT(*) AS total_records
                    FROM read_parquet('{self.data_path}')
                    WHERE {strata_col} = '{s_val}'
                    GROUP BY 1
                    ORDER BY 1 ASC
                )
                SELECT 
                    time_week,
                    events_count,
                    (total_records - events_count) AS censored_count
                FROM timeline;
            """
            df_strata = self.storage.execute_query(query)
            
            q_tot = f"SELECT COUNT(*) AS total FROM read_parquet('{self.data_path}') WHERE {strata_col} = '{s_val}'"
            total_s = int(self.storage.execute_query(q_tot).iloc[0]["total"])
            
            n_at_risk = total_s
            s_prob = 1.0
            
            for _, row in df_strata.iterrows():
                t = float(row["time_week"])
                d = int(row["events_count"])
                c = int(row["censored_count"])
                
                hazard = (d / n_at_risk) if n_at_risk > 0 else 0.0
                s_prob *= (1.0 - hazard)
                
                all_curves.append({
                    "strata_variable": strata_col,
                    "strata_group": s_val,
                    "time_week": t,
                    "n_at_risk": n_at_risk,
                    "dropouts": d,
                    "censored": c,
                    "survival_probability": round(s_prob, 4)
                })
                n_at_risk -= (d + c)

        return pd.DataFrame(all_curves)

    def compute_actuarial_life_table(self) -> pd.DataFrame:
        """
        Computes formal Actuarial Life Tables over weekly intervals [x, x+1):
        q_x = d_x / (n_x - c_x / 2)  (Effective population at risk)
        p_x = 1 - q_x
        P_x = Cumulative survival
        """
        self._ensure_data_exists()
        query = f"""
            WITH weekly_buckets AS (
                SELECT 
                    FLOOR(duration_weeks)::INT AS week_interval,
                    SUM(event_observed) AS dropouts,
                    SUM(CASE WHEN event_observed = 0 THEN 1 ELSE 0 END) AS censored
                FROM read_parquet('{self.data_path}')
                GROUP BY 1
                ORDER BY 1 ASC
            )
            SELECT week_interval, dropouts, censored FROM weekly_buckets;
        """
        df_raw = self.storage.execute_query(query)
        
        q_tot = f"SELECT COUNT(*) AS total FROM read_parquet('{self.data_path}')"
        total_pop = int(self.storage.execute_query(q_tot).iloc[0]["total"])

        life_table = []
        n_start = total_pop
        cum_survival = 1.0

        for _, row in df_raw.iterrows():
            w = int(row["week_interval"])
            d = int(row["dropouts"])
            c = int(row["censored"])

            # Half-year / half-week exposure adjustment for censored units
            effective_n = max(1.0, n_start - (c / 2.0))
            q_x = min(1.0, d / effective_n) if effective_n > 0 else 0.0
            p_x = 1.0 - q_x
            cum_survival *= p_x
            hazard_rate = d / (effective_n - (d / 2.0)) if (effective_n - (d / 2.0)) > 0 else 0.0

            life_table.append({
                "week_interval": f"Week [{w}, {w+1})",
                "interval_start": w,
                "n_entering_interval": n_start,
                "dropouts_observed": d,
                "censored_active": c,
                "effective_number_at_risk": round(effective_n, 1),
                "conditional_dropout_rate_qx": round(q_x, 4),
                "conditional_survival_rate_px": round(p_x, 4),
                "cumulative_survival_Px": round(cum_survival, 4),
                "interval_hazard_rate": round(hazard_rate, 4)
            })

            n_start -= (d + c)
            if n_start <= 0:
                break

        return pd.DataFrame(life_table)

    def compute_explainable_hazard_ratios(self) -> pd.DataFrame:
        """
        Computes Explainable Cox Proportional Hazards Hazard Ratios (HR)
        for actionable barrier interventions:
        HR = Hazard(High Risk Group) / Hazard(Baseline Group)
        Provides executive business interpretations for Tableau & C-Level.
        """
        self._ensure_data_exists()

        hypotheses = [
            {
                "barrier_name": "Tutor Feedback Latency Overload (>36h vs <=12h)",
                "high_risk_filter": "tutor_feedback_latency_hours > 36.0",
                "baseline_filter": "tutor_feedback_latency_hours <= 12.0",
                "business_action": "Enforce maximum 18h SLA on code reviews; alert academic coordinators if mentor queue backs up."
            },
            {
                "barrier_name": "Pacing & Time Deficit (<15h/week vs >=22h/week)",
                "high_risk_filter": "weekly_hours_dedicated < 15.0",
                "baseline_filter": "weekly_hours_dedicated >= 22.0",
                "business_action": "Introduce flexible pacing tracks or asynchronous weekend office hours for career switchers."
            },
            {
                "barrier_name": "Milestone Submission Lag (>4.0 days vs <=1.5 days)",
                "high_risk_filter": "assignment_lag_days > 4.0",
                "baseline_filter": "assignment_lag_days <= 1.5",
                "business_action": "Automate early proactive outreach via Slack/Email on day 3 of milestone delay."
            },
            {
                "barrier_name": "High Hurdle Difficulty Index (>3.8 vs <=2.4)",
                "high_risk_filter": "hurdle_difficulty_index > 3.8",
                "baseline_filter": "hurdle_difficulty_index <= 2.4",
                "business_action": "Decompose complex milestone projects into intermediate scaffolding deliverables."
            },
            {
                "barrier_name": "Background Disparity (Non-Technical vs STEM Graduate)",
                "high_risk_filter": "prior_experience = 'Non-Technical Career Switcher'",
                "baseline_filter": "prior_experience = 'STEM University Graduate'",
                "business_action": "Provide 2-week pre-bootcamp foundational leveling modules in Python & SQL."
            }
        ]

        hr_results = []
        for h in hypotheses:
            q_high = f"""
                SELECT 
                    SUM(event_observed) AS events,
                    SUM(duration_weeks) AS total_person_time,
                    COUNT(*) AS count_n
                FROM read_parquet('{self.data_path}')
                WHERE {h['high_risk_filter']}
            """
            df_high = self.storage.execute_query(q_high).iloc[0]
            
            q_base = f"""
                SELECT 
                    SUM(event_observed) AS events,
                    SUM(duration_weeks) AS total_person_time,
                    COUNT(*) AS count_n
                FROM read_parquet('{self.data_path}')
                WHERE {h['baseline_filter']}
            """
            df_base = self.storage.execute_query(q_base).iloc[0]

            events_h = float(df_high["events"] or 1.0)
            time_h = float(df_high["total_person_time"] or 1.0)
            hazard_h = events_h / time_h

            events_b = float(df_base["events"] or 1.0)
            time_b = float(df_base["total_person_time"] or 1.0)
            hazard_b = events_b / time_b

            hr = hazard_h / hazard_b if hazard_b > 0 else 1.0
            
            # Approximate standard error of log(HR)
            se_log_hr = np.sqrt((1.0 / events_h) + (1.0 / events_b))
            ci_lower = round(np.exp(np.log(hr) - 1.96 * se_log_hr), 2)
            ci_upper = round(np.exp(np.log(hr) + 1.96 * se_log_hr), 2)
            hr_val = round(hr, 2)

            interpretation = (
                f"Candidates exposed to this barrier face a {hr_val}x higher risk of dropout "
                f"(95% CI: [{ci_lower}, {ci_upper}]) relative to peers in the baseline condition."
            )

            hr_results.append({
                "learning_barrier": h["barrier_name"],
                "hazard_ratio": hr_val,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "high_risk_sample_size": int(df_high["count_n"]),
                "baseline_sample_size": int(df_base["count_n"]),
                "executive_interpretation": interpretation,
                "recommended_business_action": h["business_action"]
            })

        return pd.DataFrame(hr_results)

    def export_semantic_layer(self, output_dir: str = "data/semantic_layer"):
        """
        Exports full Kimball Star-Schema analytical marts in Parquet & CSV format
        for instant ingestion into Tableau Desktop and Power BI.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Dim Kaplan-Meier Overall
        df_km = self.compute_kaplan_meier_overall()
        df_km.to_parquet(os.path.join(output_dir, "dim_kaplan_meier_overall.parquet"), index=False)
        df_km.to_csv(os.path.join(output_dir, "dim_kaplan_meier_overall.csv"), index=False)

        # 2. Dim Stratified Kaplan-Meier (Background & Track)
        df_strata = self.compute_stratified_kaplan_meier("prior_experience")
        df_strata.to_parquet(os.path.join(output_dir, "dim_kaplan_meier_stratified.parquet"), index=False)
        df_strata.to_csv(os.path.join(output_dir, "dim_kaplan_meier_stratified.csv"), index=False)

        # 3. Dim Actuarial Life Table
        df_life = self.compute_actuarial_life_table()
        df_life.to_parquet(os.path.join(output_dir, "dim_actuarial_life_table.parquet"), index=False)
        df_life.to_csv(os.path.join(output_dir, "dim_actuarial_life_table.csv"), index=False)

        # 4. Dim Explainable Hazard Ratios
        df_hr = self.compute_explainable_hazard_ratios()
        df_hr.to_parquet(os.path.join(output_dir, "dim_explainable_hazard_ratios.parquet"), index=False)
        df_hr.to_csv(os.path.join(output_dir, "dim_explainable_hazard_ratios.csv"), index=False)

        # 5. Fact Student Survival Data Mart
        q_fact = f"""
            SELECT 
                candidate_id,
                cohort_id,
                track,
                prior_experience,
                weekly_hours_dedicated,
                assignment_lag_days,
                tutor_feedback_latency_hours,
                hurdle_difficulty_index,
                assessment_score_avg,
                duration_weeks,
                event_observed,
                lifecycle_status,
                primary_drop_reason,
                CASE 
                    WHEN assignment_lag_days > 4.0 OR tutor_feedback_latency_hours > 36.0 OR weekly_hours_dedicated < 14.0 THEN 'HIGH_RISK'
                    WHEN assignment_lag_days > 2.5 OR tutor_feedback_latency_hours > 20.0 THEN 'ELEVATED_RISK'
                    ELSE 'NOMINAL_TRACK'
                END AS survival_risk_stratum
            FROM read_parquet('{self.data_path}')
        """
        df_fact = self.storage.execute_query(q_fact)
        df_fact.to_parquet(os.path.join(output_dir, "fact_student_survival.parquet"), index=False)
        df_fact.to_csv(os.path.join(output_dir, "fact_student_survival.csv"), index=False)

        print(f"[Core Engine] Semantic layer exported successfully to '{output_dir}':")
        print(f"  - dim_kaplan_meier_overall ({len(df_km)} rows)")
        print(f"  - dim_kaplan_meier_stratified ({len(df_strata)} rows)")
        print(f"  - dim_actuarial_life_table ({len(df_life)} rows)")
        print(f"  - dim_explainable_hazard_ratios ({len(df_hr)} rows)")
        print(f"  - fact_student_survival ({len(df_fact):,} rows)")


def create_engine(data_path: str = "data/raw_dataset.parquet") -> AnalyticsEngine:
    """Composition Root: Instantiates clean DIP dependencies."""
    adapter = DuckDBStorageAdapter()
    return AnalyticsEngine(storage=adapter, data_path=data_path)


if __name__ == "__main__":
    engine = create_engine()
    print("==================================================================")
    print("⚡ CROSSING HURDLES: EXPLAINABLE SURVIVAL ANALYTICS ENGINE (DIP)")
    print("==================================================================")
    
    life_table = engine.compute_actuarial_life_table()
    print("\n[+] Actuarial Life Table (First 8 Weeks):")
    print(life_table[["week_interval", "n_entering_interval", "dropouts_observed", "cumulative_survival_Px", "interval_hazard_rate"]].head(8).to_string(index=False))

    hr_table = engine.compute_explainable_hazard_ratios()
    print("\n[+] Explainable Hazard Ratios & Actionable Barriers:")
    for _, r in hr_table.iterrows():
        print(f"  * {r['learning_barrier']} -> HR: {r['hazard_ratio']} (95% CI: [{r['ci_95_lower']}, {r['ci_95_upper']}])")
        print(f"    Action: {r['recommended_business_action']}")

    engine.export_semantic_layer()
