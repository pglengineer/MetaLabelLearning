import pandas as pd
import numpy as np
import os
import glob
import warnings
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# =====================================================
# 1. Technical Indicators
# =====================================================
def calculate_technical_indicators(df):
    """15分鐘資料用的技術指標（四組尺度：SS=4H、S=9H、M=24H、L=72H）"""
    tech_df = pd.DataFrame(index=df.index)

    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']
    Volume = df['Volume']

    # 通用工具
    def wilders_rma(s, period):
        return s.ewm(alpha=1/period, adjust=False).mean()

    def true_range(high, low, close):
        prev_close = close.shift(1)
        return pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

    def mfi(high, low, close, volume, period=14):
        tp = (high + low + close) / 3.0
        rmf = tp * volume
        pos = (tp > tp.shift(1)).astype(int)
        neg = (tp < tp.shift(1)).astype(int)
        pos_mf = (rmf * pos).rolling(period).sum()
        neg_mf = (rmf * neg).rolling(period).sum()
        mr = pos_mf / neg_mf
        return 100 - (100 / (1 + mr))

    def dm_di_dx(high, low, close, period=14):
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        tr = true_range(high, low, close)
        atr = wilders_rma(tr, period)
        plus_di = 100 * wilders_rma(plus_dm, period) / atr
        minus_di = 100 * wilders_rma(minus_dm, period) / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return plus_dm, minus_dm, plus_di, minus_di, dx

    # ========= TRANGE/ATR/NATR（加入 SS）=========
    # 15分鐘資料：4H=16, 14H=56, 24H=96, 72H=288
    tech_df['TRANGE'] = true_range(High, Low, Close)
    for p, tag in [(16,'SS'), (56,'S'), (96,'M'), (288,'L')]:
        atr = wilders_rma(tech_df['TRANGE'], p)
        tech_df[f'ATR_{tag}'] = atr
        tech_df[f'NATR_{tag}'] = atr / Close * 100.0

    # ========= Momentum & Oscillators =========
    # MOM / ROC / ROCR（加入 SS）
    # 15分鐘資料：4H=16, 10H=40, 24H=96, 72H=288
    for p, tag in [(16,'SS'), (40,'S'), (96,'M'), (288,'L')]:
        tech_df[f'MOM_{tag}'] = Close - Close.shift(p)
        tech_df[f'ROC_{tag}'] = Close.pct_change(p) * 100.0
        tech_df[f'ROCR_{tag}'] = Close / Close.shift(p)

    # MFI（加入 SS）
    # 15分鐘資料：4H=16, 14H=56, 24H=96, 72H=288
    for p, tag in [(16,'SS'), (56,'S'), (96,'M'), (288,'L')]:
        tech_df[f'MFI_{tag}'] = mfi(High, Low, Close, Volume, p)

    # ADX/DI/DM/DX（只保留 MINUS_DM）
    # 15分鐘資料：4H=16, 14H=56, 24H=96, 72H=288
    for p, tag in [(16,'SS'), (56,'S'), (96,'M'), (288,'L')]:
        _, minus_dm, _, _, _ = dm_di_dx(High, Low, Close, p)
        tech_df[f'MINUS_DM_{tag}'] = minus_dm

    # ========= Volume & Accumulation =========
    # Volume_SMA_20: 20個15分鐘K線（約5小時），保持不變
    tech_df['Volume_SMA_20'] = Volume.rolling(20).mean()
    tech_df['Volume_Ratio_20'] = Volume / tech_df['Volume_SMA_20']

    return tech_df

# =====================================================
# 2. Basic Statistics (Volatility)
# =====================================================
def rogers_satchell_volatility(df, window=None):
    """計算 Rogers–Satchell Volatility"""
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位: {col}")
    
    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']
    
    term1 = np.log(High / Close) * np.log(High / Open)
    term2 = np.log(Low / Close) * np.log(Low / Open)
    rs_terms = term1 + term2
    rs_terms = rs_terms.replace([np.inf, -np.inf], np.nan)
    
    if window is None:
        rs_terms_clean = rs_terms.dropna()
        if len(rs_terms_clean) == 0:
            return np.nan
        mean_term = rs_terms_clean.mean()
        return np.sqrt(mean_term)
    else:
        mean_terms = rs_terms.rolling(window=window, min_periods=1).mean()
        return np.sqrt(mean_terms)

def rolling_skewness_kurtosis(returns, window=60):
    skewness = returns.rolling(window=window, min_periods=window).skew()
    kurtosis = returns.rolling(window=window, min_periods=window).kurt()
    return skewness, kurtosis

def drawdown_recovery_rate(prices, returns, window=None):
    if window is None:
        cummax = prices.cummax()
        drawdown = 1 - prices / cummax
        recovery_rate = returns / drawdown.replace(0, np.nan)
        max_drawdown = drawdown.cummax()
    else:
        drawdown_list = []
        recovery_rate_list = []
        max_drawdown_list = []
        
        for i in range(len(prices)):
            if i < window:
                drawdown_list.append(np.nan)
                recovery_rate_list.append(np.nan)
                max_drawdown_list.append(np.nan)
                continue
            
            window_prices = prices.iloc[i-window:i+1]
            window_returns = returns.iloc[i-window:i+1]
            window_cummax = window_prices.cummax()
            window_dd = 1 - window_prices / window_cummax
            drawdown_list.append(window_dd.iloc[-1])
            
            if window_dd.iloc[-1] > 0:
                recovery = window_returns.iloc[-1] / window_dd.iloc[-1]
            else:
                recovery = np.nan
            recovery_rate_list.append(recovery)
            max_drawdown_list.append(window_dd.max())
        
        drawdown = pd.Series(drawdown_list, index=prices.index)
        recovery_rate = pd.Series(recovery_rate_list, index=prices.index)
        max_drawdown = pd.Series(max_drawdown_list, index=prices.index)
    
    return drawdown, recovery_rate, max_drawdown

def volatility_clustering_index(returns, window=60):
    abs_returns = np.abs(returns)
    def calc_vci(x):
        if len(x) < 2: return np.nan
        autocorr = pd.Series(x).autocorr(lag=1)
        return autocorr if not pd.isna(autocorr) else 0.0
    return abs_returns.rolling(window=window, min_periods=window).apply(calc_vci, raw=False)

def volatility_clustering_garch_approximation(returns, window=60):
    def calc_garch_params(x):
        if len(x) < 20: return np.nan
        squared_returns = x ** 2
        autocorr = pd.Series(squared_returns).autocorr(lag=1)
        return autocorr if not pd.isna(autocorr) else 0.0
    return returns.rolling(window=window, min_periods=window).apply(calc_garch_params, raw=True)

def entropy_of_volume(volume, window=60):
    def calc_volume_entropy(x):
        if len(x) == 0 or x.sum() == 0: return np.nan
        p = x / x.sum()
        p = p[p > 0]
        if len(p) == 0: return 0.0
        return -np.sum(p * np.log2(p))
    return volume.rolling(window=window, min_periods=window).apply(calc_volume_entropy, raw=True)

def entropy_of_volume_binned(volume, window=60, n_bins=10):
    def calc_entropy_binned(x):
        if len(x) < n_bins: return np.nan
        try:
            bins = pd.cut(x, bins=n_bins, labels=False, duplicates='drop')
            counts = pd.Series(bins).value_counts()
            p = counts / len(bins)
            p = p[p > 0]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log2(p))
        except: return np.nan
    return volume.rolling(window=window, min_periods=window).apply(calc_entropy_binned, raw=True)

# =====================================================
# 3. Volume/Price Behavior (CUSUM, Tests)
# =====================================================
def cusum_statistic(returns, window=None):
    if window is None:
        mean_return = returns.mean()
    else:
        mean_return = returns.rolling(window=window).mean()
    deviation = returns - mean_return
    cusum_stat = deviation.cumsum()
    cusum_change = cusum_stat.diff().abs()
    return cusum_stat, cusum_change

def cusum_filter_events(returns, threshold, window=None):
    _, cusum_change = cusum_statistic(returns, window)
    events = cusum_change[cusum_change > threshold].index
    return pd.DatetimeIndex(events)

def csw_cusum_test(log_prices, n_start=1, alpha=0.05):
    y = log_prices.values
    n = len(y)
    b_alpha = {0.05: 4.6, 0.01: 5.2, 0.10: 4.0}.get(alpha, 4.6)
    cusum_stats = []
    break_flags = []
    
    for t in range(n_start + 1, n):
        delta_y = np.diff(y[1:t+1])
        sigma_sq = np.var(delta_y) if len(delta_y) > 0 else 1.0
        sigma_t = np.sqrt(sigma_sq) if sigma_sq > 0 else 1.0
        y_t = y[t]
        y_n = y[n_start]
        t_minus_n = t - n_start
        
        if sigma_t > 0 and t_minus_n > 0:
            s_n_t = (y_t - y_n) / (sigma_t * np.sqrt(t_minus_n))
        else:
            s_n_t = 0
        cusum_stats.append(s_n_t)
        c_alpha = np.sqrt(b_alpha + np.log(t_minus_n))
        break_flags.append(1 if abs(s_n_t) > c_alpha else 0)
    
    index = log_prices.index[n_start + 1:]
    return pd.Series(cusum_stats, index=index), pd.Series(break_flags, index=index)

def sadf_test(log_prices, min_window=20, max_window=None):
    y = log_prices.values
    n = len(y)
    if max_window is None: max_window = n
    sadf_scores = []
    
    for t in range(min_window, n):
        max_df = -np.inf
        for t0 in range(1, t - min_window + 1):
            y_sub = y[t0:t+1]
            if len(y_sub) < min_window: continue
            try:
                delta_y = np.diff(y_sub)
                y_lag = y_sub[:-1]
                if len(delta_y) < 5: continue
                X = y_lag.reshape(-1, 1)
                model = OLS(delta_y, X).fit()
                coeff = model.params[0]
                std_err = model.bse[0] if model.bse[0] > 0 else 1.0
                df_stat = coeff / std_err
                max_df = max(max_df, df_stat)
            except: continue
        sadf_scores.append(max_df if max_df > -np.inf else 0)
    
    sadf_stat = max(sadf_scores) if sadf_scores else 0
    index = log_prices.index[min_window:]
    return sadf_stat, pd.Series(sadf_scores, index=index)

def structural_break_regime_label(market_indicator, methods=['cusum', 'csw', 'sadf']):
    regime = pd.Series(0, index=market_indicator.index)
    if 'cusum' in methods and 'CUSUM_Break' in market_indicator.columns:
        regime = regime | market_indicator['CUSUM_Break']
    if 'csw' in methods and 'CSW_Break' in market_indicator.columns:
        regime = regime | market_indicator['CSW_Break']
    if 'sadf' in methods and 'SADF_Break' in market_indicator.columns:
        regime = regime | market_indicator['SADF_Break']
    return regime

# =====================================================
# 4. Information Features
# =====================================================
def shannon_entropy_rolling(returns, window=30):
    signs = np.sign(returns)
    def calc_entropy(x):
        p_up = (x == 1).sum() / len(x)
        p_down = (x == -1).sum() / len(x)
        p_zero = (x == 0).sum() / len(x)
        entropy = 0
        if p_up > 0: entropy -= p_up * np.log2(p_up)
        if p_down > 0: entropy -= p_down * np.log2(p_down)
        if p_zero > 0: entropy -= p_zero * np.log2(p_zero)
        return entropy
    return signs.rolling(window=window, min_periods=window).apply(calc_entropy, raw=True)

def lempel_ziv_entropy(sequence):
    if isinstance(sequence, pd.Series): seq = sequence.values
    else: seq = np.array(sequence)
    binary_seq = (seq > 0).astype(int)
    n = len(binary_seq)
    if n == 0: return 0
    i = 0
    c = 0
    substrings = set()
    while i < n:
        j = i + 1
        found = False
        while j <= n:
            substring = tuple(binary_seq[i:j])
            if substring not in substrings:
                substrings.add(substring)
                c += 1
                i = j
                found = True
                break
            j += 1
        if not found: i += 1
    return (c * np.log2(n)) / n if n > 0 else 0

def lempel_ziv_entropy_rolling(returns, window=60):
    lz_series = []
    for i in range(len(returns)):
        if i < window:
            lz_series.append(np.nan)
            continue
        window_returns = returns.iloc[i-window:i]
        lz_val = lempel_ziv_entropy(window_returns)
        lz_series.append(lz_val)
    return pd.Series(lz_series, index=returns.index)

def hurst_exponent_rs(series, max_lag=None):
    n = len(series)
    if max_lag is None: max_lag = n // 4
    if max_lag < 2: return 0.5
    lags = []
    rs_values = []
    for lag in range(2, min(max_lag, n // 2)):
        n_segments = n // lag
        if n_segments < 2: continue
        rs_list = []
        for i in range(n_segments):
            segment = series.iloc[i*lag:(i+1)*lag]
            if len(segment) < 2: continue
            mean_seg = segment.mean()
            cumdev = (segment - mean_seg).cumsum()
            R = cumdev.max() - cumdev.min()
            S = segment.std()
            if S > 0: rs_list.append(R / S)
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))
    if len(lags) < 2: return 0.5
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    valid = np.isfinite(log_lags) & np.isfinite(log_rs)
    if valid.sum() < 2: return 0.5
    slope, _ = np.polyfit(log_lags[valid], log_rs[valid], 1)
    return slope

def hurst_exponent_rolling(series, window=240, max_lag=None):
    hurst_series = []
    for i in range(len(series)):
        if i < window:
            hurst_series.append(np.nan)
            continue
        window_series = series.iloc[i-window:i]
        if len(window_series) < 10:
            hurst_series.append(np.nan)
            continue
        try:
            H = hurst_exponent_rs(window_series, max_lag=max_lag)
            hurst_series.append(H)
        except:
            hurst_series.append(np.nan)
    return pd.Series(hurst_series, index=series.index)

def hurst_exponent_dfa(series, window=240):
    def dfa_analysis(x):
        n = len(x)
        if n < 10: return 0.5
        y = np.cumsum(x - x.mean())
        scales = np.logspace(1, np.log10(n//4), 10).astype(int)
        scales = scales[scales < n//4]
        scales = scales[scales > 1]
        if len(scales) < 2: return 0.5
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 2: continue
            f_list = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                x_seg = np.arange(len(segment))
                coeffs = np.polyfit(x_seg, segment, 1)
                trend = np.polyval(coeffs, x_seg)
                detrended = segment - trend
                f_list.append(np.sqrt(np.mean(detrended**2)))
            if f_list: fluctuations.append(np.mean(f_list))
        if len(fluctuations) < 2: return 0.5
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        valid = np.isfinite(log_scales) & np.isfinite(log_fluct)
        if valid.sum() < 2: return 0.5
        slope, _ = np.polyfit(log_scales[valid], log_fluct[valid], 1)
        return slope
    return series.rolling(window=window, min_periods=window).apply(dfa_analysis, raw=True)

def market_efficiency_index(returns, window=30, entropy_window=30):
    entropy = shannon_entropy_rolling(returns, window=entropy_window)
    volatility = returns.rolling(window=window).std()
    return entropy * volatility

# =====================================================
# 5. Market Microstructure
# =====================================================
def roll_spread_estimate(returns, window=60):
    def calc_roll_spread(x):
        if len(x) < 2: return np.nan
        autocorr = x.autocorr(lag=1)
        if pd.isna(autocorr) or autocorr >= 0: return 0.0
        return 2 * np.sqrt(-autocorr)
    return returns.rolling(window=window, min_periods=window).apply(calc_roll_spread, raw=False)

def amihud_illiquidity(returns, volume, window=60):
    volume_safe = volume.replace(0, np.nan)
    amihud_daily = np.abs(returns) / volume_safe
    return amihud_daily.rolling(window=window, min_periods=1).mean()

def kyle_lambda(returns, volume, window=60):
    signed_volume = np.sign(returns) * volume
    temp_df = pd.DataFrame({'returns': returns, 'signed_volume': signed_volume})
    kyle_lambda_series = []
    for i in range(len(returns)):
        if i < window:
            kyle_lambda_series.append(np.nan)
            continue
        window_data = temp_df.iloc[i-window:i]
        rets = window_data['returns'].values
        signed_vols = window_data['signed_volume'].values
        valid = ~(np.isnan(rets) | np.isnan(signed_vols))
        rets_clean = rets[valid]
        signed_vols_clean = signed_vols[valid]
        if len(rets_clean) < 10:
            kyle_lambda_series.append(np.nan)
            continue
        cov = np.cov(rets_clean, signed_vols_clean)[0, 1]
        var = np.var(signed_vols_clean)
        if var == 0 or np.isnan(cov) or np.isnan(var):
            kyle_lambda_series.append(np.nan)
        else:
            kyle_lambda_series.append(cov / var)
    return pd.Series(kyle_lambda_series, index=returns.index)

def order_flow_imbalance(returns, volume=None, window=30):
    ofi_basic = np.sign(returns)
    if volume is not None:
        signed_volume = np.sign(returns) * volume
        ofi = signed_volume.rolling(window=window, min_periods=1).mean()
    else:
        ofi = ofi_basic.rolling(window=window, min_periods=1).mean()
    return ofi

def volume_imbalance_ratio(returns, volume, window=60):
    temp_df = pd.DataFrame({'returns': returns, 'volume': volume})
    vir_series = []
    for i in range(len(returns)):
        if i < window:
            vir_series.append(np.nan)
            continue
        window_data = temp_df.iloc[i-window:i]
        rets = window_data['returns'].values
        vols = window_data['volume'].values
        v_up = vols[rets > 0].sum()
        v_down = vols[rets < 0].sum()
        total = v_up + v_down
        if total == 0:
            vir_series.append(0.0)
        else:
            vir_series.append((v_up - v_down) / total)
    return pd.Series(vir_series, index=returns.index)

def realized_spread_price_impact(prices, returns, delta_minutes=5):
    direction = np.sign(returns)
    future_prices = prices.shift(-delta_minutes)
    price_change = future_prices - prices
    realized_spread = 2 * price_change * direction
    price_impact = price_change * direction
    return realized_spread, price_impact

# =====================================================
# Main Processing Function
# =====================================================
def process_all_commodities(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        output_filename = f"{os.path.splitext(filename)[0]}_features.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename}: Output already exists at {output_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            rename_map = {
                'bidopen': 'Open', 'bidhigh': 'High', 'bidlow': 'Low', 'bidclose': 'Close', 'volume': 'Volume',
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'date': 'Date', 'datetime': 'Date'
            }
            df.rename(columns=rename_map, inplace=True)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if 'Open' in missing_cols and 'Close' not in missing_cols:
                     df['Open'] = df['Close'].shift(1).fillna(df['Close'])
                     if 'High' in missing_cols: df['High'] = df['Close']
                     if 'Low' in missing_cols: df['Low'] = df['Close']
                else:
                    print(f"Skipping {filename}: Missing columns {missing_cols}")
                    continue
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif df.index.dtype == 'object':
                df.index = pd.to_datetime(df.index)

            # Pre-calculate common series
            log_price = np.log(df['Close'])
            returns = df['Close'].pct_change()
            
            # 1. Technical Indicators
            print("  Calculating Technical Indicators...")
            tech_df = calculate_technical_indicators(df)
            
            # 2. Basic Statistics
            print("  Calculating Basic Statistics...")
            stats_df = pd.DataFrame(index=df.index)
            stats_df['RS_Volatility_20'] = rogers_satchell_volatility(df, window=20)
            stats_df['RS_Volatility_60'] = rogers_satchell_volatility(df, window=60)
            stats_df['RS_Volatility_240'] = rogers_satchell_volatility(df, window=240)
            stats_df['Skewness_60'], stats_df['Kurtosis_60'] = rolling_skewness_kurtosis(returns, window=60)
            dd_60, rr_60, mdd_60 = drawdown_recovery_rate(df['Close'], returns, window=60)
            stats_df['Drawdown_60'] = dd_60
            stats_df['Recovery_Rate_60'] = rr_60
            stats_df['VCI_60'] = volatility_clustering_index(returns, window=60)
            stats_df['Volume_Entropy_60'] = entropy_of_volume(df['Volume'], window=60)
            
            # 3. Volume/Price Behavior
            print("  Calculating Volume/Price Behavior...")
            market_df = pd.DataFrame(index=df.index)
            cusum_stat, cusum_change = cusum_statistic(returns, window=100)
            market_df['CUSUM_Stat'] = cusum_stat
            market_df['CUSUM_Change'] = cusum_change
            csw_stat, csw_break = csw_cusum_test(log_price, n_start=100)
            market_df['CSW_Stat'] = csw_stat
            market_df['CSW_Break'] = csw_break
            # SADF is slow, use smaller window or skip if too slow
            # sadf_stat, sadf_scores = sadf_test(log_price, min_window=60, max_window=240)
            # market_df['SADF_Score'] = sadf_scores
            
            # 4. Information Features
            print("  Calculating Information Features...")
            entropy_df = pd.DataFrame(index=df.index)
            entropy_df['Shannon_Entropy_60'] = shannon_entropy_rolling(returns, window=60)
            entropy_df['LZ_Entropy_60'] = lempel_ziv_entropy_rolling(returns, window=60)
            entropy_df['Hurst_RS_240'] = hurst_exponent_rolling(log_price, window=240, max_lag=60)
            entropy_df['MEI_60'] = market_efficiency_index(returns, window=60)
            
            # 5. Market Microstructure
            print("  Calculating Market Microstructure...")
            micro_df = pd.DataFrame(index=df.index)
            micro_df['Roll_Spread_60'] = roll_spread_estimate(returns, window=60)
            micro_df['Amihud_60'] = amihud_illiquidity(returns, df['Volume'], window=60)
            micro_df['Kyle_Lambda_60'] = kyle_lambda(returns, df['Volume'], window=60)
            micro_df['OFI_Volume_60'] = order_flow_imbalance(returns, volume=df['Volume'], window=60)
            micro_df['VIR_60'] = volume_imbalance_ratio(returns, df['Volume'], window=60)
            micro_df['Realized_Spread_5min'], micro_df['Price_Impact_5min'] = realized_spread_price_impact(df['Close'], returns, delta_minutes=5)

            # Combine all
            final_df = pd.concat([tech_df, stats_df, market_df, entropy_df, micro_df], axis=1)
            
            # Save
            output_filename = f"{os.path.splitext(filename)[0]}_features.csv"
            output_path = os.path.join(output_dir, output_filename)
            final_df.to_csv(output_path)
            print(f"Saved features to {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Users\a124a\OneDrive\桌面\策略開發\MetaLabelLearning\MetaNewHigh\minute_prices"
    output_directory = r"C:\Users\a124a\OneDrive\桌面\策略開發\MetaLabelLearning\MetaNewHigh\feature_data"
    
    process_all_commodities(input_directory, output_directory)
