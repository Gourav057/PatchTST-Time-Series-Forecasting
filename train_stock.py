import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train import train_patchtst

def train_stock_models():
    data_files = [
        "data/IN_Bharti_Airtel.csv",
        "data/IN_Reliance_Industries.csv",
        "data/US_Apple_Inc.csv"
    ]
    
    for df_path in data_files:
        if os.path.exists(df_path):
            print(f"\n📈 Fine-Tuning PatchTST specifically on Stock Market Dataset: {df_path}")
            train_patchtst(
                data_path=df_path,
                window_len=336,
                forecast_len=96,
                patch_len=16,
                patch_stride=8,
                d_model=128,
                n_heads=8,
                e_layers=3,
                epochs=20,
                batch_size=16,
                lr=5e-4,
                patience=5,
                save_dir="checkpoints"
            )

if __name__ == "__main__":
    train_stock_models()
