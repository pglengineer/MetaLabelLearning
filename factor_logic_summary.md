# Factor Logic and Definition Summary

This document summarizes the logical definitions for "Time Dimension Factors" and "High-Frequency Price-Volume Factors".

## I. Time Dimension Factors
**Scope**: Stock Index Futures (e.g., NAS100)
**Core Concept**: **Duration**, the time length required for the cumulative change in price or volume to reach a certain threshold.

### 1. Price Duration
*   **Definition**: The time length ($t_j - t_i$) required for the absolute price change $|P_j - P_i|$ to exceed a set threshold (`PRICE_THRESHOLD`) from time $t_i$ to $t_j$.
*   **Purpose**: Measures the speed of market price changes and the persistence of trends.
*   **Derived Factors (10 items)**:
    1.  **Price_duration_max_min**: Difference between the daily maximum and minimum Duration.
    2.  **Price_duration_vol**: Standard deviation of the daily Duration.
    3.  **Price_duration_averg**: Average of the daily Duration.
    4.  **Price_duration_corr**: Correlation coefficient between Close price and Duration.
    5.  **Duration_price** (Lagged Return): The future return (usually lagged) of the price associated with the longest Duration.
    6.  **Price_duration_slope**: Linear regression slope of the Duration series (trend term).
    7.  **Price_duration_extreme**: Proportion of extreme Durations (greater than mean + 2 standard deviations).
    8.  **Price_duration_open_close**: Average Duration of the late session (last 30 mins) minus average Duration of the early session (first 30 mins).
    9.  **Price_duration_adj_vol**: Duration volatility divided by Price volatility.
    10. **Price_volume_persistence_ratio**: Ratio of volume during high persistence periods (Duration > 80th percentile) to total daily volume.

### 2. Volume Duration
*   **Definition**: The time length required for the cumulative volume $|V_j - V_i|$ to exceed a set threshold (`VOLUME_THRESHOLD`) from time $t_i$ to $t_j$.
*   **Purpose**: Measures the speed of market liquidity exchange.
*   **Derived Factors (10 items)**:
    *   Same structure as Price Duration, including Max-Min, Vol, Averg, Corr, Return, Slope, Extreme, Open-Close, Adj Vol, Persistence Ratio.

### 3. Behavior Timestamp Factors
*   **Definition**: The specific time points when key market behaviors (e.g., highest price, max volume) occur.
*   **Normalization**: Maps time points to the $[-1, 1]$ interval (-1 is the start of the session, 1 is the end).
*   **Sessions**: Pre-market, Regular Trading, After-hours.
*   **Factor List (22 items)**:
    *   **time_to_high**: Time to reach the daily high price.
    *   **time_to_low**: Time to reach the daily low price.
    *   **time_to_max_volume**: Time to reach the maximum minute volume.
    *   **time_to_max_turnover**: Time to reach the maximum minute turnover.
    *   **time_to_close_above_open**: Time when price first crosses above the open price.
    *   **time_to_close_below_open**: Time when price first crosses below the open price.
    *   **time_to_half_volume**: Time when cumulative volume reaches 50% of the daily total.
    *   **afternoon_high_time**: Time of the new high in the afternoon session (12:00-16:00).
    *   *(The first 7 items are calculated separately for three sessions, totaling 7x3 = 21, plus afternoon_high_time = 22 factors)*

---

## II. High-Frequency Price-Volume Factors
**Scope**: Stocks and Futures
**Core Concept**: Statistical characteristics, distributions, and flow based on minute-level data.

### 1. Return Distribution Factors
*   **Logic**: Describes the statistical shape of intraday returns.
*   **Factors (8 items)**:
    1.  **RV (Realized Variance)**: Realized variance, $\sum r_i^2$.
    2.  **RSkew (Realized Skewness)**: Measures the asymmetry of the return distribution.
    3.  **RKurt (Realized Kurtosis)**: Measures the tail thickness (probability of extreme events).
    4.  **DownVolRatio**: Downside volatility ratio, $\frac{\sum_{r_i<0} r_i^2}{RV}$.
    *   *(Each of the above 4 factors includes a 20-day moving average version)*

### 2. Volume Distribution Factors
*   **Logic**: Divides the day into multiple fixed time windows (e.g., bins of 30 minutes) and calculates the ratio of volume in that window to the total daily volume.
*   **Factors (96 items)**:
    *   **vol_bin_0 ~ vol_bin_47**: Volume ratio for the 0th to 47th time windows.
    *   *(Corresponding 20-day moving average versions)*
*   **Purpose**: Captures specific intraday trading patterns (e.g., opening volume surge, close rally).

### 3. Price-Volume Correlation Factors
*   **Logic**: Calculates the correlation coefficient between intraday minute price and volume.
*   **Factors (2 items)**:
    1.  **PriceVolumeCorr**: $\text{Corr}(Price_t, Volume_t)$. Positive values indicate price increases with volume (trend); negative values indicate price increases with shrinking volume (divergence).
    2.  **PriceVolumeCorr_ma20**: 20-day moving average.

### 4. Flow Factors
*   **Logic**: Combines price direction with volume/open interest to simulate net capital inflow/outflow.
*   **Factors (2-4 items)**:
    1.  **OrderFlowVolume**: Volume-based flow.
        *   Formula: $\frac{\sum (\text{Volume} \cdot \text{Price} \cdot \text{sign}(\Delta P))}{\sum (\text{Volume} \cdot \text{Price})}$
    2.  **OrderFlowOI** (Requires Open Interest data): Flow based on Open Interest changes.
        *   Formula: $\frac{\sum (|\Delta \text{OI}| \cdot \text{Price} \cdot \text{sign}(\Delta P))}{\sum (|\Delta \text{OI}| \cdot \text{Price})}$

### 5. Momentum/Trend Factors
*   **Logic**: Measures the efficiency of price movement.
*   **Factors (1 item)**:
    1.  **TrendStrength**: $\frac{\text{Close} - \text{Open}}{\sum |\Delta P_i|}$.
        *   Values close to 1 indicate a smooth one-way trend; values close to 0 indicate oscillation. Usually includes a 20-day moving average.

---

## Summary
These factors focus on market microstructure:
1.  **Time Dimension** focuses on "Speed" and "Timing" (When & How Fast).
2.  **HFT Price-Volume** focuses on "Distribution Shape" and "Price-Volume Correlation" (Distribution & Correlation).

Generated factors are typically standardized using **Rolling Z-score (252 days)** and then analyzed for IC with future returns to generate trading signals.
