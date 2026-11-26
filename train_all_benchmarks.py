import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train import train_patchtst

DATASETS = [
    ("data/DOMAIN_Weather_Multivariate.csv", [24, 48, 96, 192, 336, 720]),
    ("data/DOMAIN_Energy_Grid_Demand.csv", [24, 48, 96, 192, 336, 720]),
    ("data/IN_Reliance_Industries.csv", [24, 48, 96]),
    ("data/IN_Bharti_Airtel.csv", [24, 48, 96]),
    ("data/IN_Infosys_Ltd.csv", [24, 48, 96]),
    ("data/US_Apple_Inc.csv", [24, 48, 96]),
    ("data/US_Nvidia_Corp.csv", [24, 48, 96])
]

def run_benchmark_suite():
    print("==================================================")
    print("🚀 TRAINING & BENCHMARKING PATCHTST ACROSS DATASETS")
    print("==================================================\n")

    all_results = []
    
    for data_path, horizons in DATASETS:
        if os.path.exists(data_path):
            dataset_name = os.path.basename(data_path).replace(".csv", "")
            print(f"\n▶ Training Dataset: {dataset_name} for Horizons: {horizons}")
            
            for H in horizons:
                res = train_patchtst(
                    data_path=data_path,
                    window_len=336,
                    forecast_len=H,
                    patch_len=16,
                    patch_stride=8,
                    d_model=128,
                    n_heads=8,
                    e_layers=3,
                    epochs=10,
                    batch_size=32,
                    lr=1e-3,
                    patience=3
                )
                all_results.append({
                    "Dataset": dataset_name,
                    "Horizon (H)": H,
                    "Lookback (L)": 336,
                    "Test MSE": res["test_mse"],
                    "Test MAE": res["test_mae"],
                    "Checkpoint": res["checkpoint"]
                })

    df_res = pd.DataFrame(all_results)
    out_file = "results/full_benchmark_summary.csv"
    os.makedirs("results", exist_ok=True)
    df_res.to_csv(out_file, index=False)

    print("\n==================================================")
    print("🏆 FULL BENCHMARK SUMMARY")
    print("==================================================")
    print(df_res.to_string(index=False))
    print(f"\nBenchmark results saved to -> {out_file}\n")

if __name__ == "__main__":
    run_benchmark_suite()
