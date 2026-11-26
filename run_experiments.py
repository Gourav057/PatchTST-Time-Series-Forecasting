import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train import train_patchtst

def run_all_horizons(
    data_path: str = "data/weather_multivariate.csv",
    horizons = [96, 192, 336, 720],
    epochs: int = 5,
    batch_size: int = 32
):
    print("==================================================")
    print("🚀 STARTING PATCHTST EMPIRICAL ABLATION TESTBENCH")
    print(f"Target Horizons: {horizons}")
    print("==================================================\n")

    summary_results = []
    
    for H in horizons:
        print(f"\n▶ Training PatchTST for Horizon H = {H} steps...")
        res = train_patchtst(
            data_path=data_path,
            window_len=336,
            forecast_len=H,
            patch_len=16,
            patch_stride=8,
            d_model=128,
            n_heads=8,
            e_layers=3,
            epochs=epochs,
            batch_size=batch_size,
            patience=3
        )
        summary_results.append({
            "Horizon (H)": f"{H} steps",
            "Lookback (L)": "336 steps",
            "Test MSE": res["test_mse"],
            "Test MAE": res["test_mae"],
            "Checkpoint": res["checkpoint"]
        })

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join("results", "ablation_summary.csv")
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print("\n==================================================")
    print("🏆 FINAL ABLATION TESTBENCH SUMMARY")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to -> {summary_path}\n")

if __name__ == "__main__":
    data_file = "data/weather_multivariate.csv"
    if not os.path.exists(data_file):
        from src.download_data import generate_weather_dataset
        generate_weather_dataset()
        
    run_all_horizons(data_path=data_file, horizons=[96, 192, 336, 720], epochs=3)
