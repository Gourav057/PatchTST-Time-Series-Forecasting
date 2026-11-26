# Comprehensive Theoretical Guide: Datasets, Features, and Variables
## B.Tech Minor Project: Multivariate Time-Series Forecasting with PatchTST

---

## 1. What are Independent and Dependent Variables? (The Basics)

In everyday mathematics and Machine Learning (ML):
* **Independent Variable ($X$)**: The input, evidence, or cause. It is the data you already know and feed into the system.
* **Dependent Variable ($Y$)**: The output, result, or effect. It is the value you want to predict, and its outcome *depends* on the independent variables.

### The Everyday Analogy
| Scenario | Independent Variable ($X$) | Dependent Variable ($Y$) |
| :--- | :--- | :--- |
| **Cooking** | Ingredients & baking temperature | Taste and texture of cake |
| **Exam** | Hours studied & revision tests taken | Exam grade |
| **Real Estate** | Square footage, bedrooms, city | House selling price |

---

## 2. The Big Shift: How Time-Series Forecasting is Different

In standard ML (like predicting house prices), columns are permanently divided:
* Column 1 (Size) = $X$
* Column 2 (Bedrooms) = $X$
* Column 3 (Price) = $Y$

**In Time-Series Forecasting, this is NOT how it works.**
Instead of dividing by columns, **we divide by TIME (the arrow of time)**:

1. **The Past is $X$ (Independent Variables)**:
   * Any data point that happened in the past (the lookback window of **336 hours**) is an independent variable. It is known historical fact.
2. **The Future is $Y$ (Dependent Variables)**:
   * Any data point that happens in the future (the forecast horizon of **96 hours**) is a dependent variable. It is unknown and depends on past history.

> **Crucial Viva Insight:**
> Can the **exact same column** (e.g., `Close` stock price) be both an independent and a dependent variable?
> **YES!** 
> - Bharti Airtel's `Close` price for the **past 336 hours** is an **Independent Variable ($X$)**.
> - Bharti Airtel's `Close` price for the **next 96 hours** is the **Dependent Variable ($Y$)**.
> This mathematical relationship is known as **Auto-Regressive Dependency** (the future depends on its own past).

---

## 3. Dataset Anatomy: Rows and Columns

### A. Number of Rows in the Datasets
| Dataset Category | File Name Examples | Row Count | Time Span / Meaning |
| :--- | :--- | :--- | :--- |
| **Intraday Stock Data (1-Hour candles)** | `IN_Bharti_Airtel_1h.csv`<br>`IN_Reliance_Industries_1h.csv`<br>`US_Apple_Inc_1h.csv` | **~5,030 to 5,070 rows** | ~2 full years of hourly trading sessions (market open to close). |
| **Daily Stock Data** | `IN_Reliance_Industries.csv`<br>`IN_Bharti_Airtel.csv` | **~745 rows** | ~3 years of daily closing prices (~250 trading days/year). |
| **Physical Benchmarks** | `DOMAIN_Weather_Multivariate.csv`<br>`DOMAIN_Energy_Grid_Demand.csv` | **5,000 rows** | 5,000 consecutive hourly sensor and electrical grid readings. |

---

### B. Number of Columns in the Datasets
When raw stock data is downloaded from Yahoo Finance (`yfinance`), it contains **6 raw columns** (1 timestamp + 5 market metrics).
Before training, our quantitative feature engineering pipeline (`add_technical_indicators`) computes **15 mathematical indicators**, giving a total of **20 numerical channels (features)** plus **1 timestamp**.

```
Total Columns = 1 Datetime (Index) + 20 Numerical Features (Channels)
```

---

## 4. Complete Breakdown of the 20 Numerical Features

### Category 1: Raw Market Price & Volume (5 Columns)
These are the foundational metrics recorded on the stock exchange floor:
1. **`Open`**: The price at which the stock traded at the beginning of the 1-hour interval.
2. **`High`**: The maximum price the stock reached during that hour (buying pressure peak).
3. **`Low`**: The minimum price the stock fell to during that hour (selling pressure trough).
4. **`Close`**: The final agreed trade price at the end of that hour. *(Most critical price reference).*
5. **`Volume`**: The total number of shares bought and sold during that hour. Indicates market liquidity and institutional commitment.

---

### Category 2: Returns & Stationarity (2 Columns)
Raw stock prices wander unpredictably (random walk). Transformers learn much better from percentage rate-of-change:
6. **`Pct_Returns`** ($\frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$):
   * Percentage price change from the previous hour. Centers the signal around $0.0$.
7. **`Log_Returns`** ($\ln(\text{Close}_t / \text{Close}_{t-1})$):
   * Continuously compounded returns. Preferred in quantitative finance because it is mathematically additive over multiple time periods.

---

### Category 3: Trend & Momentum Indicators (6 Columns)
These indicators tell the AI whether the market is in an uptrend, downtrend, or shifting direction:
8. **`EMA_12`** (Exponential Moving Average over 12 hours):
   * Fast-reacting average price that places higher statistical weight on recent hours.
9. **`EMA_26`** (Exponential Moving Average over 26 hours):
   * Medium-term baseline average price (~4 trading days).
10. **`EMA_50`** (Exponential Moving Average over 50 hours):
    * Long-term trend baseline (~7.5 trading days).
11. **`MACD`** ($\text{EMA}_{12} - \text{EMA}_{26}$):
    * Moving Average Convergence Divergence. When MACD crosses above 0, upward momentum is accelerating.
12. **`MACD_Signal`** (9-hour EMA of MACD):
    * Smoothed trigger line. When MACD crosses above its Signal line, it confirms a bullish momentum wave.
13. **`MACD_Hist`** ($\text{MACD} - \text{MACD\_Signal}$):
    * The momentum spread. Shows whether buying/selling momentum is strengthening or weakening.

---

### Category 4: Range & Oscillator Indicators (5 Columns)
These metrics measure whether the price is stretched too far from its normal equilibrium:
14. **`RSI_14`** (Relative Strength Index over 14 hours):
    * Oscillates between $0$ and $100$. Values $>70$ indicate "overbought" (due for a pullback); values $<30$ indicate "oversold" (due for a rebound).
15. **`Bollinger_Upper`** ($\text{SMA}_{20} + 2\sigma$):
    * 2 standard deviations above the 20-hour moving average. Represents statistical ceiling.
16. **`Bollinger_Lower`** ($\text{SMA}_{20} - 2\sigma$):
    * 2 standard deviations below the 20-hour moving average. Represents statistical floor.
17. **`Bollinger_Width`** ($\frac{\text{Upper} - \text{Lower}}{\text{SMA}_{20}}$):
    * Measures volatility squeeze. A narrowing band warns that an explosive breakout is imminent.
18. **`Stoch_K`** ($\frac{\text{Close} - \text{Low}_{14}}{\text{High}_{14} - \text{Low}_{14}} \times 100$):
    * Stochastic Oscillator. Measures where the current price closed relative to its 14-hour high-low price range.
19. **`Stoch_D`** (3-hour moving average of `%K`):
    * Smoothed signal line for the Stochastic oscillator.

---

### Category 5: Risk & Market Turbulence (1 Column)
20. **`Volatility_20`** ($\sigma_{20}$):
    * 20-hour rolling standard deviation of closing prices. Quantifies the uncertainty and price swing intensity of the stock.

---

## 5. Mathematical Representation in PyTorch

When the dataset is loaded into the model by PyTorch's `DataLoader`:

### Independent Variable Tensor ($X$)
```
Shape of X = [Batch_Size, Lookback_Window, Channels]
           = [32, 336, 20]
```
* **32**: Number of parallel window samples processed at once (Batch Size $B$).
* **336**: Past 336 hours of historical readings (Lookback Length $L \approx 54$ trading days).
* **20**: All 20 numerical features measured at every single hour.

### Dependent Variable Tensor ($Y$)
```
Shape of Y = [Batch_Size, Forecast_Horizon, Channels]
           = [32, 96, 20]
```
* **96**: Future 96 hours of values the model must predict (Forecast Horizon $H \approx 15$ trading days).
* **20**: All 20 future channel trajectories predicted simultaneously.

---

## 6. How PatchTST Works on These Variables

1. **RevIN (Reversible Instance Normalization)**:
   * Strips the mean and standard deviation from the input window $X$ so the Transformer sees normalized zero-mean data.
2. **Channel-Independence**:
   * Instead of mixing all 20 channels together (which causes cross-channel noise and overfitting), PatchTST splits the input into **20 separate 1D series**.
   * The **same Transformer encoder** weights process each channel independently.
3. **Subseries Patching ($P=16, S=8$)**:
   * Instead of passing individual numbers (which have low semantic meaning), it extracts sliding patches of 16 consecutive hours, stepping forward 8 hours at a time:
     $$\text{Num\_Patches } N = \left\lfloor\frac{336 - 16}{8}\right\rfloor + 1 = 41 \text{ patches}$$
4. **Self-Attention & Forecasting**:
   * The 3-layer Transformer Encoder attends across these 41 patches.
   * A Linear Projection Head maps the latent representations directly to the future **96 steps**.
   * RevIN restores the original mean and variance to yield realistic prices.

---

## 7. Viva / Examination Ready Summary

* **Q: What are the independent variables?**
  * *A: The past 336 hourly observations across all 20 numerical features (past price, volume, moving averages, RSI, and volatility).*
* **Q: What are the dependent variables?**
  * *A: The future 96 hourly observations across the same channels, with primary target being the future `Close` price.*
* **Q: Why are there 20 columns instead of just 1 (Close price)?**
  * *A: Single prices lack momentum and volatility context. By providing 20 quantitative features (returns, EMAs, MACD, RSI, Bollinger Bands), the model captures multi-faceted market dynamics.*
* **Q: How many rows are in the dataset?**
  * *A: Intraday 1-hour datasets contain ~5,030 rows (~2 years of hourly trading). Daily datasets contain ~745 rows (~3 years of trading days).*
