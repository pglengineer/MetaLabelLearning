"""
Rolling meta-label training script rebuilt from L1-MetaLabel notebook.

This module replicates the original workflow with a cleaner, fully scripted
pipeline:

1. Load and preprocess NAS100 price data
2. Apply noise filtering (daily volatility + symmetric CUSUM)
3. Generate L1 breakout signals for event detection
4. Apply triple-barrier meta-labeling to obtain bins
5. Compute sample weights (concurrency, uniqueness, time decay, class balance)
6. Engineer technical and fractional-differentiated features
7. Filter features via correlation / variance / ANOVA
8. Train ML models via a configurable PredictionPipeline
9. Perform yearly rolling training (train <= prior year, test current year)
10. Save each model's five-year prediction DataFrame

All chart titles/text are kept in English per project convention.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DataConfig:
    raw_data_path: Path = Path("equity_prices/NAS100.csv")
    train_start: pd.Timestamp = pd.Timestamp("2014-01-01")
    first_test_year: int = 2020
    test_years: int = 5
    cusum_threshold: float = 1e-3
    daily_vol_span: int = 500
    l1_entry_param: float = 1.0
    l1_window: int = 96
    meta_pt: float = 0.25
    meta_sl: float = 0.25
    meta_num_periods: int = 32
    min_label_pct: float = 0.05
    fracdiff_d: float = 0.5
    fracdiff_thres: float = 1e-3
    embargo_pct: float = 0.01
    plots: bool = False


@dataclass
class ModelConfig:
    name: str
    estimator: BaseEstimator
    param_grid: Dict[str, List[Any]]
    use_scaler: bool = True


@dataclass
class RollingConfig:
    models: List[ModelConfig]
    output_dir: Path = Path("predictions")
    save_intermediate: bool = True


DATA_CFG = DataConfig()
ROLLING_CFG = RollingConfig(
    models=[
        ModelConfig(
            "lightgbm",
            HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=7,
                learning_rate=0.05,
                l2_regularization=0.0,
                random_state=42,
            ),
            param_grid={
                "max_depth": [5, 7, 9],
                "learning_rate": [0.03, 0.05, 0.08],
                "max_leaf_nodes": [31, 63],
            },
            use_scaler=False,
        ),
        ModelConfig(
            "xgboost",
            HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=7,
                learning_rate=0.05,
                random_state=42,
            ),
            param_grid={
                "max_depth": [5, 7, 9],
                "learning_rate": [0.03, 0.05, 0.08],
                "max_iter": [200, 400],
            },
            use_scaler=False,
        ),
        ModelConfig(
            "rf",
            RandomForestClassifier(
                n_estimators=500, max_depth=12, n_jobs=-1, random_state=42
            ),
            param_grid={
                "max_depth": [10, 12, 15],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 4, 6],
            },
            use_scaler=False,
        ),
        ModelConfig(
            "logreg",
            LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
            ),
            param_grid={
                "C": [0.01, 0.1, 1.0, 5.0],
            },
            use_scaler=True,
        ),
    ],
    output_dir=Path("predictions"),
)


# ============================================================================
# Utility functions
# ============================================================================


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex.")
    return df


def load_price_data(config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(config.raw_data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    df["Close"] = df["BidClose"]
    df["Open"] = df["BidOpen"]
    df["High"] = df["BidHigh"]
    df["Low"] = df["BidLow"]
    df["Volume"] = df["Volume"].fillna(0.0)
    mask = (df.index >= config.train_start) & (
        df.index < config.train_start + relativedelta(years=config.test_years + 10)
    )
    df = df.loc[mask]
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ============================================================================
# Noise filtering
# ============================================================================


class NoiseFilter:
    def __init__(self, vol_span: int) -> None:
        self.vol_span = vol_span
        self.daily_vol: Optional[pd.Series] = None

    def calculate_daily_vol(self, close: pd.Series) -> pd.Series:
        close = close.asfreq("15T", method="pad")
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(close.index[df0 - 1], index=close.index[-len(df0) :])
        r = close.loc[df0.index] / close.loc[df0.values].values - 1
        daily_vol = r.ewm(span=self.vol_span).std()
        self.daily_vol = daily_vol.dropna()
        return self.daily_vol

    def cusum_filter(self, close: pd.Series, threshold: float) -> pd.DatetimeIndex:
        diff = close.diff().dropna()
        s_pos, s_neg = 0.0, 0.0
        events = []
        for ts, change in diff.items():
            s_pos = max(0.0, s_pos + change)
            s_neg = min(0.0, s_neg + change)
            if s_pos > threshold:
                s_pos = 0.0
                events.append(ts)
            elif s_neg < -threshold:
                s_neg = 0.0
                events.append(ts)
        return pd.DatetimeIndex(events)


# ============================================================================
# Strategy signals
# ============================================================================


class L1Strategy:
    def __init__(self, entry_param: float, rolling_window: int) -> None:
        self.entry_param = entry_param
        self.rolling_window = rolling_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        hh = (
            df["High"]
            .shift(1)
            .rolling(self.rolling_window, min_periods=self.rolling_window)
            .max()
        )
        ll = (
            df["Low"]
            .shift(1)
            .rolling(self.rolling_window, min_periods=self.rolling_window)
            .min()
        )
        df["hl_range"] = hh - ll
        df["breakout_price"] = ll + self.entry_param * df["hl_range"]
        df["signal"] = np.where(df["Close"] > df["breakout_price"], 1, 0)
        df["side"] = np.where(df["Close"] > df["breakout_price"], 1, -1)
        df["side"] = df["side"].shift(1).fillna(0)
        return df

    def get_events(self, signals: pd.DataFrame) -> pd.DatetimeIndex:
        entries = signals.index[signals["signal"] == 1]
        return pd.DatetimeIndex(entries)


# ============================================================================
# Meta labeling (triple barrier)
# ============================================================================


class MetaLabeling:
    def __init__(
        self,
        pt_sl: Tuple[float, float],
        num_periods: int,
        min_label_pct: float,
    ) -> None:
        self.pt, self.sl = pt_sl
        self.num_periods = num_periods
        self.min_label_pct = min_label_pct

    def get_vertical_barriers(
        self, events: pd.DatetimeIndex, close: pd.Series
    ) -> pd.Series:
        t1 = {}
        for t in events:
            t1[t] = close.index[close.index.get_loc(t) + self.num_periods - 1]
        return pd.Series(t1)

    def get_events(
        self,
        close: pd.Series,
        t_events: pd.DatetimeIndex,
        daily_vol: pd.Series,
        side: pd.Series,
    ) -> pd.DataFrame:
        t1 = self.get_vertical_barriers(t_events, close)
        events = pd.DataFrame(index=t_events)
        events["t1"] = t1
        events["trgt"] = daily_vol.loc[t_events].fillna(daily_vol.median())
        events["pt"] = self.pt
        events["sl"] = self.sl
        events["side"] = side.loc[t_events].fillna(0)
        return events.dropna()

    def apply_triple_barrier(
        self, close: pd.Series, events: pd.DataFrame
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=events.index)
        for start, row in events.iterrows():
            end = row["t1"]
            path = close.loc[start:end]
            if len(path) == 0:
                continue
            ret = path / close.loc[start] - 1
            top = row["pt"] * row["trgt"]
            bottom = -row["sl"] * row["trgt"]
            touched = 0
            for ts, r in ret.items():
                if r > top:
                    touched = 1
                    end = ts
                    break
                if r < bottom:
                    touched = -1
                    end = ts
                    break
            out.loc[start, "t1"] = end
            out.loc[start, "ret"] = ret.loc[end]
            out.loc[start, "bin"] = touched
            out.loc[start, "side"] = row["side"]
        out["bin"].fillna(0, inplace=True)
        return out.dropna()

    def drop_rare_labels(self, bins: pd.DataFrame) -> pd.DataFrame:
        while True:
            counts = bins["bin"].value_counts(normalize=True)
            if counts.min() >= self.min_label_pct or len(counts) <= 2:
                break
            target = counts.idxmin()
            bins = bins[bins["bin"] != target]
        return bins


# ============================================================================
# Sample weights
# ============================================================================


class SampleWeightEngine:
    def compute_num_co_events(
        self, close_index: pd.DatetimeIndex, t1: pd.Series
    ) -> pd.Series:
        count = pd.Series(0.0, index=close_index)
        for start, end in t1.items():
            count.loc[start:end] += 1.0
        return count[count > 0]

    def compute_uniqueness(self, t1: pd.Series, num_co_events: pd.Series) -> pd.Series:
        w = pd.Series(index=t1.index)
        for start, end in t1.items():
            w.loc[start] = (1.0 / num_co_events.loc[start:end]).mean()
        return w.fillna(0)

    def apply_time_decay(self, weights: pd.Series, decay: float = 0.5) -> pd.Series:
        ranks = np.linspace(0, 1, len(weights))
        decay_factor = (1 - decay) + decay * ranks
        return weights.sort_index() * decay_factor

    def compute_class_balance(self, labels: pd.Series) -> pd.Series:
        class_w = compute_sample_weight("balanced", labels)
        return pd.Series(class_w, index=labels.index)

    def build_sample_weights(
        self,
        close_index: pd.DatetimeIndex,
        t1: pd.Series,
        labels: pd.Series,
        decay: float = 0.5,
    ) -> pd.Series:
        num_co = self.compute_num_co_events(close_index, t1)
        uniq = self.compute_uniqueness(t1, num_co)
        uniq = self.apply_time_decay(uniq, decay=decay)
        class_balance = self.compute_class_balance(labels)
        weights = uniq.mul(class_balance, fill_value=0)
        weights /= weights.mean()
        return weights.clip(lower=1e-6)


# ============================================================================
# Feature engineering
# ============================================================================


class FractionalDiff:
    def __init__(self, d: float, thres: float) -> None:
        self.d = d
        self.thres = thres

    def get_weights(self, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] * (self.d - k + 1) / k
            if abs(w_) < self.thres:
                break
            w.append(w_)
        return np.array(w[::-1]).reshape(-1, 1)

    def transform(self, series: pd.Series) -> pd.Series:
        weights = self.get_weights(len(series))
        width = len(weights)
        vals = series.values
        diff = np.empty(len(series))
        diff[: width - 1] = np.nan
        for i in range(width - 1, len(series)):
            window = vals[i - width + 1 : i + 1]
            diff[i] = np.dot(weights.T, window)[0]
        return pd.Series(diff, index=series.index, name=f"{series.name}_fracdiff")


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"]).diff()
    for win in [5, 10, 20, 60, 120]:
        df[f"sma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ema_{win}"] = df["Close"].ewm(span=win).mean()
        df[f"vol_{win}"] = df["Close"].pct_change().rolling(win).std()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["mfi_14"] = compute_mfi(df)
    df["obv"] = compute_obv(df)
    df["atr_14"] = compute_atr(df, 14)
    return df.dropna()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=close.index).rolling(period).mean()
    loss = pd.Series(loss, index=close.index).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typ = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typ * df["Volume"]
    pos = money_flow.where(typ.diff() > 0, 0.0)
    neg = money_flow.where(typ.diff() < 0, 0.0).abs()
    mfr = pos.rolling(period).sum() / neg.rolling(period).sum()
    return 100 - (100 / (1 + mfr))


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_feature_matrix(df: pd.DataFrame, fracdiff: FractionalDiff) -> pd.DataFrame:
    tech = calculate_technical_indicators(df)
    frac_cols = {}
    for col in ["Close", "Volume"]:
        frac_cols[f"{col}_fracdiff"] = fracdiff.transform(df[col])
    frac_df = pd.DataFrame(frac_cols)
    feature_df = pd.concat([tech, frac_df], axis=1)
    return feature_df.dropna()


# ============================================================================
# Feature filtering
# ============================================================================


class FeatureFilteringPipeline:
    def __init__(self, corr_threshold: float = 0.95, var_threshold: float = 0.0):
        self.corr_threshold = corr_threshold
        self.var_threshold = var_threshold
        self.var_filter = VarianceThreshold(var_threshold)

    def correlation_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [
            col
            for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]
        return X.drop(columns=drop_cols, errors="ignore")

    def variance_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == 0:
            return X
        self.var_filter.fit(X)
        mask = self.var_filter.get_support()
        return X.loc[:, mask]

    def anova_filter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 50,
    ) -> pd.DataFrame:
        if X.shape[1] <= top_n:
            return X
        scores = {}
        for col in X.columns:
            groups = [
                X[col][y == label].dropna().values
                for label in sorted(y.unique())
                if len(X[col][y == label].dropna()) > 30
            ]
            if len(groups) < 2:
                continue
            stat, pvalue = f_oneway(*groups)
            scores[col] = -np.log10(pvalue + 1e-12)
        if not scores:
            return X
        selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
        return X[selected]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = self.correlation_filter(X)
        X = self.variance_filter(X)
        X = self.anova_filter(X, y)
        return X


# ============================================================================
# Modeling pipeline
# ============================================================================


class PredictionPipeline:
    def __init__(
        self,
        model_cfg: ModelConfig,
        scoring: str = "f1_macro",
    ) -> None:
        self.model_cfg = model_cfg
        self.scoring = scoring
        self.pipeline: Optional[Pipeline] = None

    def _clone_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        estimator = self.model_cfg.estimator.__class__(**self.model_cfg.estimator.get_params())
        estimator.set_params(**params)
        return estimator

    def build(self, params: Dict[str, Any]) -> Pipeline:
        steps: List[Tuple[str, Any]] = []
        if self.model_cfg.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append((self.model_cfg.name, self._clone_estimator(params)))
        return Pipeline(steps)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "PredictionPipeline":
        best_score = -np.inf
        best_params = {}
        for params in ParameterGrid(self.model_cfg.param_grid):
            pipe = self.build(params)
            fit_params = {}
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            pipe.fit(X, y, **fit_params)
            preds = pipe.predict(X)
            score = f1_score(y, preds, average="macro")
            if score > best_score:
                best_score = score
                best_params = params
        self.pipeline = self.build(best_params)
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.pipeline.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted.")
        preds = pd.Series(self.pipeline.predict(X), index=X.index)
        proba = self.pipeline.predict_proba(X)
        return preds, proba


# ============================================================================
# Rolling training orchestration
# ============================================================================


def prepare_datasets(
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    raw = load_price_data(config)
    nf = NoiseFilter(config.daily_vol_span)
    daily_vol = nf.calculate_daily_vol(raw["Close"])
    events_idx = nf.cusum_filter(raw["Close"], config.cusum_threshold)

    strategy = L1Strategy(config.l1_entry_param, config.l1_window)
    signal_df = strategy.generate_signals(raw.loc[events_idx])
    t_events = strategy.get_events(signal_df)

    meta = MetaLabeling(
        (config.meta_pt, config.meta_sl),
        config.meta_num_periods,
        config.min_label_pct,
    )
    events = meta.get_events(raw["Close"], t_events, daily_vol, signal_df["side"])
    bins = meta.apply_triple_barrier(raw["Close"], events)
    bins = meta.drop_rare_labels(bins)

    frac = FractionalDiff(config.fracdiff_d, config.fracdiff_thres)
    feature_df = build_feature_matrix(raw, frac)

    aligned_idx = bins.index.intersection(feature_df.index)
    X = feature_df.loc[aligned_idx]
    y = bins.loc[aligned_idx, "bin"].astype(int)
    t1 = bins.loc[aligned_idx, "t1"]
    events_aligned = events.loc[aligned_idx]

    swe = SampleWeightEngine()
    sample_weight = swe.build_sample_weights(raw.index, t1, y)

    selector = FeatureFilteringPipeline()
    X_filtered = selector.fit_transform(X, y)
    sample_weight = sample_weight.loc[X_filtered.index]
    y = y.loc[X_filtered.index]
    events_aligned = events_aligned.loc[X_filtered.index]
    return X_filtered, y, sample_weight, events_aligned


def run_single_year(
    model_cfg: ModelConfig,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    events: pd.DataFrame,
    config: DataConfig,
    year: int,
) -> pd.DataFrame:
    train_end = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=1)
    test_start = pd.Timestamp(year=year, month=1, day=1)
    test_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(minutes=15)
    train_mask = (X.index >= config.train_start) & (X.index <= train_end)
    test_mask = (X.index >= test_start) & (X.index <= test_end)
    if test_mask.sum() == 0:
        raise ValueError(f"No test samples for {year}.")

    pipeline = PredictionPipeline(model_cfg)
    pipeline.fit(
        X.loc[train_mask],
        y.loc[train_mask],
        sample_weight.loc[train_mask],
    )
    preds, proba = pipeline.predict(X.loc[test_mask])

    pred_df = pd.DataFrame(
        {
            "t1": events.loc[preds.index, "t1"],
            "side": events.loc[preds.index, "side"],
            "true_label": y.loc[test_mask],
            "predicted_label": preds,
            "prediction_confidence": proba.max(axis=1),
        },
        index=preds.index,
    )
    for idx, class_id in enumerate(pipeline.pipeline.classes_):
        pred_df[f"proba_{class_id}"] = proba[:, idx]
    pred_df["model"] = model_cfg.name
    pred_df["train_range"] = f"{config.train_start.date()}_{train_end.date()}"
    pred_df["test_year"] = year
    pred_df["is_correct"] = pred_df["true_label"] == pred_df["predicted_label"]
    return pred_df


def rolling_training_loop(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    events: pd.DataFrame,
    data_cfg: DataConfig,
    rolling_cfg: RollingConfig,
) -> Dict[str, pd.DataFrame]:
    rolling_cfg.output_dir.mkdir(exist_ok=True)
    results = {}
    for model_cfg in rolling_cfg.models:
        yearly_frames = []
        for offset in range(data_cfg.test_years):
            year = data_cfg.first_test_year + offset
            pred_df = run_single_year(
                model_cfg,
                X,
                y,
                sample_weight,
                events,
                data_cfg,
                year,
            )
            yearly_frames.append(pred_df)
        model_pred = pd.concat(yearly_frames).sort_index()
        save_path = rolling_cfg.output_dir / (
            f"pred_{model_cfg.name}_{data_cfg.first_test_year}_"
            f"{data_cfg.first_test_year + data_cfg.test_years - 1}.csv"
        )
        model_pred.to_csv(save_path)
        results[model_cfg.name] = model_pred
    return results


def summarize_predictions(pred_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "samples": int(len(pred_df)),
        "accuracy": float(pred_df["is_correct"].mean()),
        "macro_f1": float(f1_score(pred_df["true_label"], pred_df["predicted_label"], average="macro")),
        "confusion_matrix": confusion_matrix(pred_df["true_label"], pred_df["predicted_label"]).tolist(),
    }


def main() -> None:
    print("🚀 Rebuilding pipeline from notebook ...")
    X, y, sample_weight, events = prepare_datasets(DATA_CFG)
    preds = rolling_training_loop(
        X,
        y,
        sample_weight,
        events,
        DATA_CFG,
        ROLLING_CFG,
    )
    summary = {model: summarize_predictions(df) for model, df in preds.items()}
    ROLLING_CFG.output_dir.mkdir(exist_ok=True)
    (ROLLING_CFG.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Rolling training complete. Results saved to predictions/.")


if __name__ == "__main__":
    main()

