"""
src/data_generator.py - Generador de Datos Sintéticos Realistas (50,000+ registros)
Modela la física estocástica del dominio de Crossing Hurdles para Causal & Survival Lifecycle Analytics.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

def generate_synthetic_dataset(num_records: int = 50000, output_path: str = "data/raw_dataset.parquet") -> pd.DataFrame:
    print(f"[Data Generator] Generating {num_records:,} records for Crossing Hurdles...")
    start_time = time.time()
    
    np.random.seed(42)
    record_ids = np.arange(100001, 100001 + num_records)
    categories = ["Segment_A", "Segment_B", "Segment_C", "Enterprise_Tier"]
    
    df = pd.DataFrame({
        "transaction_id": record_ids,
        "entity_id": np.random.randint(1000, 5000, size=num_records),
        "segment": np.random.choice(categories, size=num_records, p=[0.4, 0.3, 0.2, 0.1]),
        "base_cost": np.round(np.random.lognormal(mean=3.5, sigma=0.6, size=num_records), 2),
        "operational_metric": np.random.normal(loc=45.0, scale=12.0, size=num_records),
        "event_timestamp": pd.date_range(start="2025-01-01", periods=num_records, freq="T"),
        "status": np.random.choice(["COMPLETED", "PENDING", "ANOMALY", "REVIEW"], size=num_records, p=[0.85, 0.08, 0.04, 0.03])
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"[Data Generator] Generated {len(df):,} records in {elapsed:.2f}s -> {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/raw_dataset.parquet")
    args = parser.parse_args()
    generate_synthetic_dataset(args.records, args.output)
