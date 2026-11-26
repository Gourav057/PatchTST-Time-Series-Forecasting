# PatchTST Multivariate Time-Series Forecasting System

An implementation of the **PatchTST** methodology ([ICLR 2023 Paper](https://arxiv.org/abs/2211.14730)) targeting financial markets (`yfinance`) and global weather metrics.

---

## 🌟 Key Architecture Features

1. **Channel Independence (CI)**:
   Treats each univariate time-series channel independently through a shared Transformer Encoder backbone, preventing inter-channel noise during feature extraction.

2. **Subseries Patching (`unfold`)**:
   Segments lookback time-series $L$ into overlapping patches of length $P$ with stride $S$:
   $$\text{Num Patches } N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1$$
   Tensor shape transformation: $[B, L, C] \to [B, C, N, P] \to [B \times C, N, D]$.

3. **Reversible Instance Normalization (RevIN)**:
   Normalizes sequence data per-instance before input and reverses scaling on forecast output to handle distribution shift.

4. **Self-Attention Probing**:
   Extracts layer-wise multi-head self-attention weight matrices $[B \times C, N, N]$ for patch-level interpretability.

5. **Interactive Dashboard**:
   Streamlit interface (`app.py`) featuring forecast horizon selection ($96, 192, 336, 720$), Plotly interactive time-series plots, Seaborn attention heatmaps, and tensor shape flow inspector.

---

## 📁 Directory Structure

```
patchtst_forecasting/
├── app.py                      # Streamlit interactive forecasting application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── data/                       # Dataset directory (CSV files)
│   └── README.md
├── src/                        # Core PyTorch source code
│   ├── __init__.py
│   ├── data_loader.py          # StandardScaler, TimeSeriesDataset, channel_independent_patching
│   └── model.py                # RevIN, PatchTSTEncoder, PatchTST PyTorch model
├── notebooks/                  # Jupyter notebooks for EDA and training
│   └── README.md
├── dashboard/                  # Dashboard custom assets & components
│   └── README.md
└── tests/                      # Automated unit tests
    └── test_components.py
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python3 tests/test_components.py
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Forecast Horizons Supported
- **96 steps** (~4 days hourly or ~3 months daily)
- **192 steps** (~8 days hourly)
- **336 steps** (2 weeks hourly)
- **720 steps** (1 month hourly)
