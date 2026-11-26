import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure src modules are resolvable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import create_dataloaders
from src.model import PatchTST

def train_patchtst(
    data_path: str,
    window_len: int = 336,
    forecast_len: int = 96,
    patch_len: int = 16,
    patch_stride: int = 8,
    d_model: int = 128,
    n_heads: int = 8,
    e_layers: int = 3,
    d_ff: int = 256,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    save_dir: str = "checkpoints",
    results_dir: str = "results",
    device: str = "cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing PatchTST Training on device: {device}")
    
    # 1. Load Data
    print(f"📥 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    if "Close" in df.columns and "Open" in df.columns:
        from src.data_loader import add_technical_indicators
        df = add_technical_indicators(df)
        
    # Filter numeric columns (excluding Date/Timestamp)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].dropna()
    
    print(f"  Dataset Shape: {df_numeric.shape} (Channels C={df_numeric.shape[1]})")

    # 2. Build DataLoaders
    train_loader, val_loader, test_loader, scaler, meta = create_dataloaders(
        df_numeric,
        window_len=window_len,
        forecast_len=forecast_len,
        patch_len=patch_len,
        patch_stride=patch_stride,
        batch_size=batch_size
    )

    num_channels = meta["num_channels"]
    num_patches = meta["num_patches"]
    print(f"  Configuration: Lookback L={window_len}, Horizon H={forecast_len}, Patches N={num_patches}, Patch P={patch_len}")

    # 3. Model Initialization
    model = PatchTST(
        c_in=num_channels,
        target_window=forecast_len,
        seq_len=window_len,
        patch_len=patch_len,
        stride=patch_stride,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=0.1,
        revin=True
    ).to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Early Stopping variables & Asset-specific checkpoint filename
    asset_name = os.path.splitext(os.path.basename(data_path))[0]
    checkpoint_path = os.path.join(save_dir, f"patchtst_{asset_name}_H{forecast_len}.pt")
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n🔥 Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_mse_loss = 0.0
        train_mae_loss = 0.0
        for batch in train_loader:
            x_batch = batch["x"].to(device)  # [B, L, C]
            y_batch = batch["y"].to(device)  # [B, H, C]

            optimizer.zero_grad()
            y_pred = model(x_batch)         # [B, H, C]
            
            loss_mse = criterion_mse(y_pred, y_batch)
            loss_mae = criterion_mae(y_pred, y_batch)
            
            loss_mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_mse_loss += loss_mse.item() * len(x_batch)
            train_mae_loss += loss_mae.item() * len(x_batch)

        scheduler.step()
        train_mse = train_mse_loss / len(train_loader.dataset)
        train_mae = train_mae_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_mse_loss = 0.0
        val_mae_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch["x"].to(device)
                y_batch = batch["y"].to(device)
                y_pred = model(x_batch)

                val_mse_loss += criterion_mse(y_pred, y_batch).item() * len(x_batch)
                val_mae_loss += criterion_mae(y_pred, y_batch).item() * len(x_batch)

        val_mse = val_mse_loss / len(val_loader.dataset)
        val_mae = val_mae_loss / len(val_loader.dataset)

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f} | Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

        # Checkpoint Saving & Early Stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_channels": num_channels,
                    "window_len": window_len,
                    "forecast_len": forecast_len,
                    "patch_len": patch_len,
                    "patch_stride": patch_stride,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "e_layers": e_layers,
                    "d_ff": d_ff
                },
                "val_mse": val_mse,
                "val_mae": val_mae
            }, checkpoint_path)
            print(f"  💾 Saved Best Model Checkpoint -> {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  🛑 Early stopping triggered after {epoch} epochs.")
                break

    # 5. Final Evaluation on Test Set
    print("\n📊 Evaluating Best Checkpoint on Test Set...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_mse_loss = 0.0
    test_mae_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch["x"].to(device)
            y_batch = batch["y"].to(device)
            y_pred = model(x_batch)

            test_mse_loss += criterion_mse(y_pred, y_batch).item() * len(x_batch)
            test_mae_loss += criterion_mae(y_pred, y_batch).item() * len(x_batch)

    test_mse = test_mse_loss / len(test_loader.dataset)
    test_mae = test_mae_loss / len(test_loader.dataset)

    results = {
        "dataset": data_path,
        "window_len": window_len,
        "forecast_len": forecast_len,
        "test_mse": round(float(test_mse), 5),
        "test_mae": round(float(test_mae), 5),
        "val_best_mse": round(float(best_val_loss), 5),
        "checkpoint": checkpoint_path
    }

    results_file = os.path.join(results_dir, f"metrics_H{forecast_len}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n==========================================")
    print(f"🎉 FINAL TEST RESULTS (Horizon H={forecast_len}):")
    print(f"   Test MSE: {test_mse:.5f}")
    print(f"   Test MAE: {test_mae:.5f}")
    print(f"   Metrics saved -> {results_file}")
    print("==========================================\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTST Model Training & Evaluation Engine")
    parser.add_argument("--data_path", type=str, default="data/weather_multivariate.csv")
    parser.add_argument("--window_len", type=int, default=336)
    parser.add_argument("--forecast_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()

    train_patchtst(
        data_path=args.data_path,
        window_len=args.window_len,
        forecast_len=args.forecast_len,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )
