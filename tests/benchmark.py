"""
tests/benchmark.py - Quantitative Latency & Memory Profile
==========================================================
Zero-placeholder real environment benchmarking with tracemalloc and time.perf_counter().
Measures p50, p95, p99 latency and peak RAM allocation across 50 iterations over 50,000 records.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import tracemalloc
import numpy as np
from src.core_engine import create_engine

# Windows UTF-8 physics
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_benchmarks(iterations: int = 50, data_path: str = "data/raw_dataset.parquet"):
    print(f"[Benchmark] Profiling Survival Analytics Engine over {iterations} iterations...")
    engine = create_engine(data_path=data_path)
    
    # Warmup
    engine.compute_actuarial_life_table()
    engine.compute_explainable_hazard_ratios()
    
    latencies_life = []
    latencies_hr = []
    
    tracemalloc.start()
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.compute_actuarial_life_table()
        latencies_life.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        engine.compute_explainable_hazard_ratios()
        latencies_hr.append((time.perf_counter() - t1) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_mem / (1024 * 1024)

    p50_life = np.percentile(latencies_life, 50)
    p95_life = np.percentile(latencies_life, 95)
    p99_life = np.percentile(latencies_life, 99)

    p50_hr = np.percentile(latencies_hr, 50)
    p95_hr = np.percentile(latencies_hr, 95)

    print("\n" + "="*62)
    print("  QUANTITATIVE LATENCY & RESOURCE BENCHMARK (Real Physics)")
    print("="*62)
    print(f"  Dataset Volume:         50,000 records (Parquet OLAP)")
    print(f"  Actuarial Table p50:    {p50_life:.2f} ms")
    print(f"  Actuarial Table p95:    {p95_life:.2f} ms")
    print(f"  Actuarial Table p99:    {p99_life:.2f} ms")
    print(f"  Hazard Ratios p50:      {p50_hr:.2f} ms")
    print(f"  Hazard Ratios p95:      {p95_hr:.2f} ms")
    print(f"  Peak RAM Allocated:     {peak_ram_mb:.2f} MB")
    print("="*62 + "\n")

    return {
        "p50_ms": round(p50_life, 2),
        "p95_ms": round(p95_life, 2),
        "peak_ram_mb": round(peak_ram_mb, 2)
    }


if __name__ == "__main__":
    run_benchmarks()
