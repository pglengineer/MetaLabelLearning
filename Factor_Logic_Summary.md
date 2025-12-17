# Factor Logic and Definition Summary

This document summarizes the logical definitions for the 9 major factor categories extracted from the project.

## 1. Fractional Differentiation
**Core Concept**: Resolves the issue of memory loss caused by traditional differencing, preserving long-term memory while achieving stationarity.
*   **Factors**:
    1.  **FracDiff_d0.5**: Fractional differentiation with order $d=0.5$.
    2.  **FracDiff_d0.75**: Fractional differentiation with order $d=0.75$.
    3.  **FracDiff_d1.0**: Standard first difference ($d=1.0$), equivalent to price changes.

## 2. Technical Indicators
**Core Concept**: Includes classic indicators to capture price patterns across multiple scales (SS=4H, S=14H, M=24H, L=72H).
*   **Trend & Momentum**:
    1.  **MOM**: Momentum ($Close_t - Close_{t-p}$).
    2.  **ROC**: Rate of Change (Percentage change).
    3.  **ROCR**: Rate of Change Ratio.
    4.  **MINUS_DM**: Directional Movement (Down).
*   **Volatility & Range**:
    1.  **TRANGE**: True Range.
    2.  **ATR**: Average True Range (Wilder's RMA).
    3.  **NATR**: Normalized ATR.
*   **Volume**:
    1.  **MFI**: Money Flow Index.
    2.  **Volume_SMA**: Simple Moving Average of Volume.
    3.  **Volume_Ratio**: Ratio of current Volume to SMA.

## 3. Volatility Indicators
**Core Concept**: Captures the level of market panic and the intensity of price fluctuations.
*   **Factors**:
    1.  **RS_Volatility (Rogers-Satchell)**: Drift-independent volatility estimator using OHLC data. (Overall and Rolling Windows: 20, 60, 240, 1440).

## 4. Market Indicators
**Core Concept**: Reflects the overall breadth, depth, and regime of the market (Structural Breaks & Bubbles).
*   **Regime Shift (CUSUM)**:
    1.  **CUSUM_Stat**: Cumulative deviations from the mean.
    2.  **CUSUM_Break**: Signals when deviation exceeds threshold.
*   **Parameter Instability**:
    1.  **BDE_Test (Brown–Durbin–Evans)**: Recursive residuals test for coefficient changes.
    2.  **CSW_Test (Chu–Stinchcombe–White)**: Log-price steady state deviation test.
*   **Bubble Detection (Explosiveness)**:
    1.  **SADF (Supremum ADF)**: Detects a single bubble event.
    2.  **GSADF (Generalized SADF)**: Detects multiple bubble events (Double-rolling window).
    3.  **Regime Label**: Composite signal of structural instability.

## 5. Entropy & Information
**Core Concept**: Uses information theory to capture the randomness and complexity of price sequences.
*   **Information Measures**:
    1.  **Shannon_Entropy**: Randomness of return signs (+1/0/-1).
    2.  **LZ_Entropy (Lempel-Ziv)**: Compressibility/Complexity of the return sequence.
    3.  **MEI (Market Efficiency Index)**: Entropy normalized by Volatility.
*   **Fractal Dynamics**:
    1.  **Hurst_RS**: Long-term memory using Rescaled Range analysis.
    2.  **Hurst_DFA**: Long-term memory using Detrended Fluctuation Analysis.

## 6. Microstructure
**Core Concept**: Extracts insights from finer tick-level price–volume microstructure.
*   **Clustering**:
    1.  **VCI (Volatility Clustering Index)**: Autocorrelation of absolute returns.
    2.  **VCI_GARCH**: GARCH-approximated clustering.
*   **Liquidity Cost**:
    1.  **Roll_Spread**: Effective bid-ask spread estimate from serial covariance.
    2.  **Amihud_Illiquidity**: Price impact per unit of volume ($|Ret|/Vol$).
    3.  **Realized_Spread**: Temporary price reversal cost.
*   **Price Impact**:
    1.  **Kyle_Lambda**: Slope of price impact (Covariance of Price Change & Signed Volume).
    2.  **Price_Impact**: Permanent price change correlated with trade direction.

## 7. Statistical Features
**Core Concept**: Based on statistical distribution features.
*   **Distribution Shape**:
    1.  **Skewness**: Asymmetry of returns.
    2.  **Kurtosis**: Tail thickness / Extremity frequency.
*   **Drawdown Dynamics**:
    1.  **Drawdown**: Decline from peak.
    2.  **Max_Drawdown**: Maximum observed decline in window.
    3.  **Recovery_Rate**: Speed of recovery from drawdown.

## 8. HFT Indicators
**Core Concept**: Derived from high-frequency data, including buy–sell pressure imbalance and order flow information.
*   **Order Flow**:
    1.  **OFI (Order Flow Imbalance)**: Net Order Flow (Basic & Volume-Weighted).
    2.  **VIR (Volume Imbalance Ratio)**: Normalized volume imbalance.
    3.  **OrderFlowVolume**: Flow factor based on tick price changes.
*   **Intraday Distributions**:
    1.  **Return Distribution**: RV, Realized Skew/Kurt, Downside Volatility Ratio.
    2.  **Volume Bins**: Ratio of volume in specific time bins (e.g., 30-min slots) to daily total.
*   **Correlations & Trend**:
    1.  **PriceVolumeCorr**: Correlation between minute price and volume.
    2.  **TrendStrength**: Efficiency of price movement (Net Change / Total Distance).

## 9. Time Factors
**Core Concept**: Captures intraday and intraweek seasonal effects (Duration & Timestamps).
*   **Duration**:
    1.  **Price_Duration**: Time to accumulate a specific price change.
    2.  **Volume_Duration**: Time to accumulate a specific volume.
    3.  **Derived Metrics**: Volatility, Slope, Persistence Ratio of the duration series.
*   **Seasonality (Timestamps)**:
    1.  **Behavior Timestamps**: Normalized time of day for High, Low, Max Volume, Max Turnover.
    2.  **Session Logic**: Interactions between Pre-market, Regular, and After-hours behaviors (e.g., Gap vs Intraday).
