"""
RAVEN: A Regime-Aware Variable-context Expert Network
=======================================================

This single file implements a reproducible, Tushare-based version of the
RAVEN paper:

    He et al., "RAVEN: A Regime-Aware Variable-context Expert Network for
    Financial Time Series Forecasting", arXiv:2606.24062v1.

The script contains:
    1. Tushare data download and local caching;
    2. historical HS300 membership handling;
    3. data cleaning and survivorship-bias-aware filtering;
    4. explicit OHLCV/technical factor construction;
    5. causal sequence dataset construction;
    6. the RAVEN model: patch importance, CIT routing, temporal experts,
       GCR global branch, CAW correlation-aware fusion;
    7. MSE + entropy + diversity training loss;
    8. Pearson correlation, cross-sectional RankIC/ICIR and a simple
       long-only top-K backtest.

Important:
    The paper describes the input as OHLCV/spread/engineered factors but does
    not publish one unique proprietary factor list. This implementation uses a
    transparent OHLCV-based factor set so that the complete data-to-model
    pipeline is executable and auditable. It is a faithful implementation of
    the model idea and experimental protocol, not a claim that undisclosed
    private preprocessing can be recovered exactly.

Quick start:
    1. Set the local environment variable TUSHARE_TOKEN before running.
    2. Install dependencies:
       pip install tushare pandas numpy torch scikit-learn tqdm
    3. Debug with a small universe:
       python raven_tushare_reproduction.py --mode all --max-stocks 20 --epochs 3
    4. Full run:
       python raven_tushare_reproduction.py --mode all

The default local study split is:
    train: 2023-01-01 to 2024-12-31
    valid: 2025-01-01 to 2025-12-31
    test : 2026-01-01 to 2026-12-31

The 2026 test interval is evaluated only over dates returned by Tushare, so it
is a partial-year out-of-sample period until the year is complete. Future-horizon
boundary samples are purged so labels do not cross train/valid/test splits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================================
# 0. USER CONFIGURATION: READ TOKEN FROM LOCAL ENVIRONMENT
# ============================================================================

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
FEATURE_CACHE_VERSION = 2


@dataclass
class Config:
    # ---- Data source ----
    tushare_token: str = TUSHARE_TOKEN
    index_code: str = "000300.SH"
    start_date: str = "20230101"
    end_date: str = "20261231"
    train_start: str = "20230101"
    train_end: str = "20241231"
    valid_start: Optional[str] = "20250101"
    valid_end: Optional[str] = "20251231"
    test_start: str = "20260101"
    test_end: str = "20261231"
    min_listing_days: int = 180
    exclude_st: bool = True
    download_daily_basic: bool = False
    max_stocks: Optional[int] = None

    # ---- Local cache ----
    data_root: str = "./raven_tushare_data"
    raw_dirname: str = "raw"
    processed_dirname: str = "processed"
    output_dirname: str = "outputs"
    api_sleep_seconds: float = 0.15
    api_retries: int = 5

    # ---- RAVEN paper hyperparameters ----
    forecast_horizon: int = 10
    max_lookback: int = 120
    patch_len: int = 16
    embed_dim: int = 128
    num_experts: int = 3
    thresholds: Tuple[float, ...] = (0.3, 0.6, 0.9)
    expert_layers: int = 3
    global_layers: int = 1
    num_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.10
    lambda_entropy: float = 0.10
    lambda_diversity: float = 0.01

    # ---- Training ----
    seed: int = 42
    batch_size: int = 512
    epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    grad_clip: float = 1.0
    device: str = "auto"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    selection_metric: str = "valid_rankic"

    # ---- Feature normalization ----
    cross_sectional_normalize: bool = True
    winsorize_quantile: float = 0.01

    # ---- Portfolio evaluation ----
    rebalance_every: int = 10
    top_k: int = 30
    transaction_cost_bps: float = 10.0

    # ---- Optional operational controls ----
    force_download: bool = False
    force_rebuild_features: bool = False

    @property
    def root(self) -> Path:
        return Path(self.data_root)

    @property
    def raw_dir(self) -> Path:
        return self.root / self.raw_dirname

    @property
    def processed_dir(self) -> Path:
        return self.root / self.processed_dirname

    @property
    def output_dir(self) -> Path:
        return self.root / self.output_dirname

    @property
    def effective_lookback(self) -> int:
        # The paper defines N=floor(Lmax/plen). We use the most recent complete
        # patch block and drop the incomplete oldest remainder.
        return (self.max_lookback // self.patch_len) * self.patch_len

    def validate(self) -> None:
        if not self.tushare_token or "在这里" in self.tushare_token:
            raise ValueError(
                "请先在本地环境变量 TUSHARE_TOKEN 中填入你的 Tushare Token。"
            )
        if self.num_experts != len(self.thresholds):
            raise ValueError("num_experts 必须等于 thresholds 的数量。")
        if not all(0.0 < x <= 1.0 for x in self.thresholds):
            raise ValueError("thresholds 必须位于 (0, 1]。")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("thresholds 必须递增。")
        if self.patch_len <= 0 or self.max_lookback < self.patch_len:
            raise ValueError("max_lookback 必须不小于 patch_len。")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience 至少为 1。")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta 不能为负。")
        if self.selection_metric not in {"valid_loss", "valid_rankic"}:
            raise ValueError("selection_metric 只能是 valid_loss 或 valid_rankic。")
        if not 0.0 <= self.winsorize_quantile < 0.5:
            raise ValueError("winsorize_quantile 必须位于 [0, 0.5)。")

        train_start = pd.Timestamp(self.train_start)
        train_end = pd.Timestamp(self.train_end)
        valid_start = pd.Timestamp(self.valid_start) if self.valid_start else None
        valid_end = pd.Timestamp(self.valid_end) if self.valid_end else None
        test_start = pd.Timestamp(self.test_start)
        test_end = pd.Timestamp(self.test_end)
        if train_start > train_end or test_start > test_end:
            raise ValueError("训练集和测试集起止日期不合法。")
        if valid_start is None or valid_end is None:
            raise ValueError("当前训练流程要求显式设置验证集日期。")
        if valid_start > valid_end or not (train_end < valid_start <= valid_end < test_start):
            raise ValueError("时间切分必须满足 train < valid < test，且各区间不能重叠。")

        data_start = pd.Timestamp(self.start_date)
        data_end = pd.Timestamp(self.end_date)
        if data_start > data_end:
            raise ValueError("数据下载起止日期不合法。")
        if data_start > train_start or data_end < test_end:
            raise ValueError(
                "下载日期必须覆盖完整的 train / valid / test 区间；"
                "请检查 start_date、end_date 与切分日期。"
            )


# ============================================================================
# 1. UTILITIES
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dirs(cfg: Config) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cache_request(cfg: Config) -> Dict[str, Any]:
    """Fields that determine whether raw and processed caches are reusable."""
    return {
        "index_code": cfg.index_code,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "max_stocks": cfg.max_stocks,
        "download_daily_basic": cfg.download_daily_basic,
    }


def raw_cache_matches(cfg: Config) -> bool:
    """Return True only when the raw cache was built for this exact request."""
    manifest_path = cfg.raw_dir / "download_manifest.json"
    daily_path = cfg.raw_dir / "daily_all.pkl"
    if cfg.force_download or not manifest_path.exists() or not daily_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    return all(manifest.get(key) == value for key, value in cache_request(cfg).items())


def feature_cache_matches(cfg: Config) -> bool:
    manifest_path = cfg.processed_dir / "feature_manifest.json"
    if not manifest_path.exists() or not raw_cache_matches(cfg):
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    expected = {
        **cache_request(cfg),
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "cross_sectional_normalize": cfg.cross_sectional_normalize,
        "winsorize_quantile": cfg.winsorize_quantile,
        "forecast_horizon": cfg.forecast_horizon,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def date_coverage(frame: pd.DataFrame, date_col: str = "trade_date") -> Dict[str, Any]:
    """Small, serializable coverage summary used in logs and run metadata."""
    if frame.empty or date_col not in frame.columns:
        return {"rows": int(len(frame)), "first_date": None, "last_date": None, "n_dates": 0}
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if dates.empty:
        return {"rows": int(len(frame)), "first_date": None, "last_date": None, "n_dates": 0}
    return {
        "rows": int(len(frame)),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "n_dates": int(dates.nunique()),
        "n_stocks": int(frame["ts_code"].nunique()) if "ts_code" in frame.columns else None,
    }


def date_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    return pd.Timestamp(value).strftime("%Y%m%d")


def safe_corr(x: Sequence[float], y: Sequence[float], method: str = "pearson") -> float:
    a = pd.Series(np.asarray(x, dtype=float))
    b = pd.Series(np.asarray(y, dtype=float))
    mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    return float(a[mask].corr(b[mask], method=method))


def choose_device(cfg: Config) -> str:
    import torch

    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ============================================================================
# 2. TUSHARE DOWNLOAD
# ============================================================================


def create_tushare_api(cfg: Config):
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError("请先运行 pip install tushare。") from exc
    ts.set_token(cfg.tushare_token)
    return ts.pro_api(cfg.tushare_token)


def tushare_call(cfg: Config, function, **kwargs) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(cfg.api_retries):
        try:
            result = function(**kwargs)
            if result is None:
                return pd.DataFrame()
            return result.copy()
        except Exception as exc:  # Tushare may transiently reject requests.
            last_error = exc
            wait = min(30.0, (attempt + 1) * 2.0)
            print(f"[Tushare retry {attempt + 1}/{cfg.api_retries}] {exc}; wait {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Tushare API 调用失败: {kwargs}") from last_error


def fetch_stock_basic(cfg: Config, pro) -> pd.DataFrame:
    path = cfg.raw_dir / "stock_basic.pkl"
    if path.exists() and not cfg.force_download:
        return pd.read_pickle(path)

    frames: List[pd.DataFrame] = []
    fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date"
    for status in ["L", "D", "P"]:
        time.sleep(cfg.api_sleep_seconds)
        frame = tushare_call(
            cfg,
            pro.stock_basic,
            exchange="",
            list_status=status,
            fields=fields,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("stock_basic 没有返回数据，请检查 Token 权限。")

    result = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
    result.to_pickle(path)
    return result


def fetch_hs300_members(cfg: Config, pro) -> pd.DataFrame:
    """Get historical HS300 constituents where possible.

    index_member contains in/out dates for historical membership on accounts
    with the required permission. If unavailable, the fallback uses index_weight
    constituents and writes a warning because that fallback can contain
    survivorship bias.
    """

    path = cfg.raw_dir / "hs300_members.pkl"
    if path.exists() and not cfg.force_download:
        return pd.read_pickle(path)

    try:
        time.sleep(cfg.api_sleep_seconds)
        members = tushare_call(
            cfg,
            pro.index_member,
            index_code=cfg.index_code,
        )
    except Exception as exc:
        warnings.warn(f"index_member 获取失败，将尝试 index_weight 回退: {exc}")
        members = pd.DataFrame()

    historical = True
    if members.empty or "con_code" not in members.columns:
        historical = False
        time.sleep(cfg.api_sleep_seconds)
        weights = tushare_call(
            cfg,
            pro.index_weight,
            index_code=cfg.index_code,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
        )
        if weights.empty:
            raise RuntimeError("无法获取 HS300 成分股，请检查 index_member/index_weight 权限。")
        members = weights[["con_code"]].drop_duplicates().copy()
        members["in_date"] = cfg.start_date
        members["out_date"] = cfg.end_date

    members = members.copy()
    if "in_date" not in members.columns:
        members["in_date"] = cfg.start_date
    if "out_date" not in members.columns:
        members["out_date"] = cfg.end_date
    members["in_date"] = members["in_date"].fillna(cfg.start_date).astype(str)
    members["out_date"] = members["out_date"].fillna(cfg.end_date).astype(str)
    members["in_date"] = members["in_date"].str.replace("-", "", regex=False)
    members["out_date"] = members["out_date"].str.replace("-", "", regex=False)

    overlap = (members["in_date"] <= cfg.end_date) & (members["out_date"] >= cfg.start_date)
    members = members.loc[overlap].copy()
    members = members.drop_duplicates(["con_code", "in_date", "out_date"])
    members["historical_membership_available"] = historical
    members.to_pickle(path)

    if not historical:
        warnings.warn(
            "当前使用 index_weight 回退成分股列表，不能完全避免幸存者偏差；"
            "建议开通 index_member 权限后重新下载。"
        )
    return members


def fetch_one_daily(cfg: Config, pro, ts_code: str) -> pd.DataFrame:
    path = cfg.raw_dir / "daily" / f"{ts_code.replace('.', '_')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not cfg.force_download:
        try:
            return pd.read_csv(path)
        except Exception:
            path.unlink(missing_ok=True)

    time.sleep(cfg.api_sleep_seconds)
    daily = tushare_call(
        cfg,
        pro.daily,
        ts_code=ts_code,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    if daily.empty:
        return pd.DataFrame()

    daily.to_csv(path, index=False)
    return daily


def fetch_one_daily_basic(cfg: Config, pro, ts_code: str) -> pd.DataFrame:
    path = cfg.raw_dir / "daily_basic" / f"{ts_code.replace('.', '_')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not cfg.force_download:
        try:
            return pd.read_csv(path)
        except Exception:
            path.unlink(missing_ok=True)

    fields = (
        "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pb,ps,total_mv,circ_mv"
    )
    time.sleep(cfg.api_sleep_seconds)
    basic = tushare_call(
        cfg,
        pro.daily_basic,
        ts_code=ts_code,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        fields=fields,
    )
    if basic.empty:
        return pd.DataFrame()
    basic.to_csv(path, index=False)
    return basic


def download_market_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download and cache daily data for the historical HS300 universe."""

    ensure_dirs(cfg)
    pro = create_tushare_api(cfg)
    stock_basic = fetch_stock_basic(cfg, pro)
    members = fetch_hs300_members(cfg, pro)

    codes = sorted(members["con_code"].dropna().unique().tolist())
    if cfg.max_stocks is not None:
        codes = codes[: cfg.max_stocks]
    print(f"[download] universe size = {len(codes)}")

    all_daily: List[pd.DataFrame] = []
    for n, code in enumerate(codes, start=1):
        try:
            daily = fetch_one_daily(cfg, pro, code)
            if not daily.empty:
                all_daily.append(daily)
            if n % 20 == 0 or n == len(codes):
                print(f"[download] daily {n}/{len(codes)}")
        except Exception as exc:
            warnings.warn(f"跳过 {code}: {exc}")

    if not all_daily:
        raise RuntimeError("没有下载到任何 daily 数据。")
    daily = pd.concat(all_daily, ignore_index=True)

    # daily_basic is optional because its endpoint quota may be stricter.
    if cfg.download_daily_basic:
        basic_frames: List[pd.DataFrame] = []
        for n, code in enumerate(codes, start=1):
            try:
                basic = fetch_one_daily_basic(cfg, pro, code)
                if not basic.empty:
                    basic_frames.append(basic)
                if n % 20 == 0 or n == len(codes):
                    print(f"[download] daily_basic {n}/{len(codes)}")
            except Exception as exc:
                warnings.warn(f"daily_basic 跳过 {code}: {exc}")
        basic_all = pd.concat(basic_frames, ignore_index=True) if basic_frames else pd.DataFrame()
    else:
        basic_all = pd.DataFrame()

    daily.to_pickle(cfg.raw_dir / "daily_all.pkl")
    if not basic_all.empty:
        basic_all.to_pickle(cfg.raw_dir / "daily_basic_all.pkl")

    # Write this only after download finishes.  The old script wrote the
    # manifest before fetching data and could later mistake stale files for a
    # complete cache after the requested date range changed.
    manifest = {
        **cache_request(cfg),
        "codes": codes,
        "n_codes_requested": len(codes),
        "historical_membership_available": bool(
            members.get("historical_membership_available", pd.Series([False])).iloc[0]
        ),
        "actual_coverage": date_coverage(daily),
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    save_json(manifest, cfg.raw_dir / "download_manifest.json")
    print(f"[download] actual coverage = {manifest['actual_coverage']}")
    return daily, stock_basic if basic_all.empty else stock_basic


# ============================================================================
# 3. DATA CLEANING AND FACTOR CONSTRUCTION
# ============================================================================


def clean_daily_data(
    daily: pd.DataFrame,
    stock_basic: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Clean raw daily data without using future information."""

    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ts_code", "trade_date", "close"])
    df = df.drop_duplicates(["ts_code", "trade_date"])
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[(df["high"] >= df["low"]) & (df["high"] >= df["close"]) & (df["low"] <= df["close"])]
    df["vol"] = df["vol"].fillna(0.0).clip(lower=0.0)
    df["amount"] = df["amount"].fillna(0.0).clip(lower=0.0)

    meta_cols = ["ts_code", "name", "list_date", "delist_date", "industry", "market"]
    meta_cols = [c for c in meta_cols if c in stock_basic.columns]
    meta = stock_basic[meta_cols].drop_duplicates("ts_code").copy()
    df = df.merge(meta, on="ts_code", how="left")

    if cfg.exclude_st and "name" in df.columns:
        name = df["name"].fillna("").astype(str)
        df = df[~name.str.contains(r"ST|退", case=False, regex=True)]

    if cfg.min_listing_days > 0 and "list_date" in df.columns:
        list_dt = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
        age = (df["trade_date"] - list_dt).dt.days
        df = df[age.isna() | (age >= cfg.min_listing_days)]

    df = df[(df["trade_date"] >= pd.Timestamp(cfg.start_date)) & (df["trade_date"] <= pd.Timestamp(cfg.end_date))]
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    min_periods = min_periods or max(5, window // 2)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-8)


def construct_factors(cleaned: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Construct transparent causal OHLCV factors.

    Every rolling statistic uses observations available at or before t. The
    future target is created only after all features have been built.
    """

    df = cleaned.copy()
    feature_frames: List[pd.DataFrame] = []

    for code, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date").copy()
        close = g["close"].astype(float)
        open_ = g["open"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["vol"].astype(float).clip(lower=0.0)
        amount = g["amount"].astype(float).clip(lower=0.0)

        log_close = np.log(close.clip(lower=1e-8))
        ret1 = log_close.diff()
        prev_close = close.shift(1)

        f = pd.DataFrame(index=g.index)
        f["ret_1"] = ret1
        f["ret_2"] = log_close.diff(2)
        f["ret_5"] = log_close.diff(5)
        f["ret_10"] = log_close.diff(10)
        f["ret_20"] = log_close.diff(20)

        # Exclude the current day's return from momentum factors. This is the
        # standard "formation at t, prediction after t" convention.
        for w in [5, 10, 20, 60, 120]:
            f[f"mom_{w}"] = log_close.shift(1) - log_close.shift(w + 1)

        f["reversal_1"] = -ret1
        f["reversal_5"] = -log_close.diff(5)

        for w in [5, 10, 20, 60]:
            f[f"volatility_{w}"] = ret1.rolling(w, min_periods=max(3, w // 2)).std()
            f[f"volume_z_{w}"] = _rolling_zscore(np.log1p(volume), w)

        f["downside_vol_20"] = ret1.where(ret1 < 0).rolling(20, min_periods=10).std()
        f["intraday_range"] = (high - low) / (prev_close.abs() + 1e-8)
        f["gap_return"] = np.log((open_.abs() + 1e-8) / (prev_close.abs() + 1e-8))
        f["close_location"] = (close - low) / (high - low + 1e-8)
        f["hl_range_20"] = (high.rolling(20, min_periods=10).max() - low.rolling(20, min_periods=10).min()) / (close + 1e-8)
        f["price_ma20"] = close / (close.rolling(20, min_periods=10).mean() + 1e-8) - 1.0
        f["price_ma60"] = close / (close.rolling(60, min_periods=30).mean() + 1e-8) - 1.0
        f["amount_log"] = np.log1p(amount)
        f["amount_change_5"] = np.log1p(amount).diff(5)
        f["amihud_20"] = (ret1.abs() / (amount.abs() + 1e-8)).rolling(20, min_periods=10).mean()

        # Optional Tushare daily_basic features. They are included only when
        # --download_daily_basic (or the Config flag) is enabled and the API
        # returns the corresponding columns.
        for col in [
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pb",
            "ps",
            "total_mv",
            "circ_mv",
        ]:
            if col in g.columns:
                f[col] = pd.to_numeric(g[col], errors="coerce")

        # RSI-like momentum oscillator without TA-Lib dependency.
        up = ret1.clip(lower=0.0)
        down = (-ret1.clip(upper=0.0))
        avg_up = up.rolling(14, min_periods=7).mean()
        avg_down = down.rolling(14, min_periods=7).mean()
        rs = avg_up / (avg_down + 1e-8)
        f["rsi14"] = 100.0 - 100.0 / (1.0 + rs)

        # Keep raw identifiers and target inputs.
        out = g[["ts_code", "trade_date", "close"]].copy()
        out["log_close"] = log_close
        out = pd.concat([out, f], axis=1)
        out["future_log_return"] = log_close.shift(-cfg.forecast_horizon) - log_close
        feature_frames.append(out)

    factors = pd.concat(feature_frames, ignore_index=True)
    factor_cols = [c for c in factors.columns if c not in {"ts_code", "trade_date", "close", "log_close", "future_log_return"}]
    factors = factors.replace([np.inf, -np.inf], np.nan)
    factors = factors.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return factors, factor_cols


def cross_sectional_normalize_factors(
    factors: pd.DataFrame,
    factor_cols: List[str],
    cfg: Config,
) -> pd.DataFrame:
    """Winsorize and z-score each factor within a trading date.

    At date t this uses only the cross-section observable after that day's
    close. It therefore improves scale comparability without looking ahead.
    """
    if not cfg.cross_sectional_normalize:
        return factors

    result = factors.copy()
    q = cfg.winsorize_quantile
    for _, index in result.groupby("trade_date", sort=False).groups.items():
        values = result.loc[index, factor_cols]
        lower = values.quantile(q)
        upper = values.quantile(1.0 - q)
        clipped = values.clip(lower=lower, upper=upper, axis=1)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0, ddof=0).replace(0.0, np.nan)
        normalized = (clipped - mean) / std
        # A constant factor carries no cross-sectional information that day.
        # Preserve genuine missing values: replacing those with zero would make
        # an immature rolling factor look like a valid observation and let it
        # enter the training samples too early.
        constant_cols = std[std.isna()].index.tolist()
        if constant_cols:
            normalized.loc[:, constant_cols] = normalized.loc[:, constant_cols].where(
                clipped.loc[:, constant_cols].isna(), 0.0
            )
        result.loc[index, factor_cols] = normalized.where(clipped.notna())
    return result


def prepare_features(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    ensure_dirs(cfg)
    feature_path = cfg.processed_dir / "features.pkl"
    names_path = cfg.processed_dir / "feature_names.json"
    if (
        feature_path.exists()
        and names_path.exists()
        and not cfg.force_rebuild_features
        and not cfg.force_download
        and feature_cache_matches(cfg)
    ):
        factors = pd.read_pickle(feature_path)
        with names_path.open("r", encoding="utf-8") as f:
            factor_cols = json.load(f)
        return factors, factor_cols

    daily_path = cfg.raw_dir / "daily_all.pkl"
    basic_path = cfg.raw_dir / "stock_basic.pkl"
    if cfg.force_download or not raw_cache_matches(cfg) or not basic_path.exists():
        download_market_data(cfg)
    daily = pd.read_pickle(daily_path)
    stock_basic = pd.read_pickle(basic_path)
    cleaned = clean_daily_data(daily, stock_basic, cfg)
    daily_basic_path = cfg.raw_dir / "daily_basic_all.pkl"
    if cfg.download_daily_basic and daily_basic_path.exists():
        daily_basic = pd.read_pickle(daily_basic_path).copy()
        daily_basic["trade_date"] = pd.to_datetime(
            daily_basic["trade_date"], format="%Y%m%d", errors="coerce"
        )
        basic_cols = [
            "ts_code",
            "trade_date",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pb",
            "ps",
            "total_mv",
            "circ_mv",
        ]
        basic_cols = [c for c in basic_cols if c in daily_basic.columns]
        cleaned = cleaned.merge(
            daily_basic[basic_cols].drop_duplicates(["ts_code", "trade_date"]),
            on=["ts_code", "trade_date"],
            how="left",
        )
    cleaned.to_pickle(cfg.processed_dir / "cleaned_daily.pkl")
    cleaned_coverage = date_coverage(cleaned)
    print(f"[data] cleaned coverage = {cleaned_coverage}")
    factors, factor_cols = construct_factors(cleaned, cfg)
    factors = cross_sectional_normalize_factors(factors, factor_cols, cfg)
    factors.to_pickle(feature_path)
    save_json({"feature_names": factor_cols}, names_path)
    save_json(
        {
            **cache_request(cfg),
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "cross_sectional_normalize": cfg.cross_sectional_normalize,
            "winsorize_quantile": cfg.winsorize_quantile,
            "forecast_horizon": cfg.forecast_horizon,
            "actual_coverage": date_coverage(factors),
        },
        cfg.processed_dir / "feature_manifest.json",
    )
    print(f"[features] rows={len(factors):,}, factors={len(factor_cols)}, coverage={date_coverage(factors)}")
    return factors, factor_cols


# ============================================================================
# 4. LAZY SEQUENCE DATASET
# ============================================================================


@dataclass
class StockBlock:
    ts_code: str
    dates: np.ndarray
    X: np.ndarray
    y: np.ndarray
    close: np.ndarray


@dataclass
class SampleIndex:
    block_id: int
    end_idx: int


class RavenSequenceDataset:
    def __init__(
        self,
        blocks: List[StockBlock],
        indices: List[SampleIndex],
        lookback: int,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.blocks = blocks
        self.indices = indices
        self.lookback = lookback
        self.target_mean = target_mean
        self.target_std = max(float(target_std), 1e-8)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample = self.indices[item]
        block = self.blocks[sample.block_id]
        end = sample.end_idx
        start = end - self.lookback + 1
        x = block.X[start : end + 1]
        y_raw = block.y[end]
        y = (y_raw - self.target_mean) / self.target_std
        return (
            x.astype(np.float32),
            np.float32(y),
            block.ts_code,
            str(pd.Timestamp(block.dates[end]).date()),
            np.float32(y_raw),
        )


def build_blocks(factors: pd.DataFrame, factor_cols: List[str]) -> List[StockBlock]:
    blocks: List[StockBlock] = []
    for code, g in factors.groupby("ts_code", sort=True):
        g = g.sort_values("trade_date").copy()
        X = g[factor_cols].to_numpy(dtype=np.float32)
        y = g["future_log_return"].to_numpy(dtype=np.float32)
        dates = g["trade_date"].to_numpy(dtype="datetime64[ns]")
        close = g["close"].to_numpy(dtype=np.float32)
        blocks.append(StockBlock(str(code), dates, X, y, close))
    return blocks


def split_indices(
    blocks: List[StockBlock],
    cfg: Config,
) -> Tuple[List[SampleIndex], List[SampleIndex], List[SampleIndex], float, float]:
    train_indices: List[SampleIndex] = []
    valid_indices: List[SampleIndex] = []
    test_indices: List[SampleIndex] = []
    train_targets: List[float] = []
    lookback = cfg.effective_lookback

    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    test_start = pd.Timestamp(cfg.test_start)
    test_end = pd.Timestamp(cfg.test_end)
    valid_start = pd.Timestamp(cfg.valid_start) if cfg.valid_start else None
    valid_end = pd.Timestamp(cfg.valid_end) if cfg.valid_end else None
    observed_dates = np.concatenate([block.dates for block in blocks]) if blocks else np.asarray([])

    for block_id, block in enumerate(blocks):
        for end_idx in range(lookback - 1, len(block.dates)):
            date = pd.Timestamp(block.dates[end_idx])
            y = float(block.y[end_idx])
            x_window = block.X[end_idx - lookback + 1 : end_idx + 1]
            if not np.isfinite(y) or not np.isfinite(x_window).all():
                continue

            # Purge samples whose future-horizon label crosses a split
            # boundary. The target is formed with a row-based horizon within
            # each stock block, so use the corresponding future observation
            # date rather than a calendar-day approximation.
            future_idx = end_idx + cfg.forecast_horizon
            if future_idx >= len(block.dates):
                continue
            future_date = pd.Timestamp(block.dates[future_idx])
            sample = SampleIndex(block_id, end_idx)

            if (
                valid_start is not None
                and valid_end is not None
                and valid_start <= date <= valid_end
                and future_date <= valid_end
            ):
                valid_indices.append(sample)
            elif (
                train_start <= date <= train_end
                and future_date <= train_end
            ):
                train_indices.append(sample)
                train_targets.append(y)
            elif (
                test_start <= date <= test_end
                and future_date <= test_end
            ):
                test_indices.append(sample)

    if not train_indices or not valid_indices or not test_indices:
        observed_start = str(pd.Timestamp(observed_dates.min()).date()) if len(observed_dates) else "None"
        observed_end = str(pd.Timestamp(observed_dates.max()).date()) if len(observed_dates) else "None"
        raise RuntimeError(
            "样本切分为空: "
            f"train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}。"
            f"因子数据实际覆盖 {observed_start} 至 {observed_end}；"
            f"请求切分为 train[{cfg.train_start},{cfg.train_end}]、"
            f"valid[{cfg.valid_start},{cfg.valid_end}]、"
            f"test[{cfg.test_start},{cfg.test_end}]。"
            "请先确认 Tushare 返回了相应年份的数据；若改过日期范围，使用 "
            "--force-download --force-rebuild-features 重新构建缓存。"
        )
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets) + 1e-8)
    print(
        f"[split] train={len(train_indices):,}, valid={len(valid_indices):,}, "
        f"test={len(test_indices):,}, target_mean={target_mean:.6g}, target_std={target_std:.6g}"
    )
    return train_indices, valid_indices, test_indices, target_mean, target_std


def summarize_split(
    blocks: List[StockBlock],
    indices: List[SampleIndex],
) -> Dict[str, Any]:
    if not indices:
        return {"n_samples": 0, "n_stocks": 0, "first_date": None, "last_date": None}
    dates = [pd.Timestamp(blocks[item.block_id].dates[item.end_idx]) for item in indices]
    codes = {blocks[item.block_id].ts_code for item in indices}
    return {
        "n_samples": len(indices),
        "n_stocks": len(codes),
        "first_date": str(min(dates).date()),
        "last_date": str(max(dates).date()),
    }


def collate_raven(batch):
    import torch

    xs, ys, codes, dates, raw_y = zip(*batch)
    return (
        torch.from_numpy(np.stack(xs)),
        torch.tensor(ys, dtype=torch.float32),
        list(codes),
        list(dates),
        torch.tensor(raw_y, dtype=torch.float32),
    )


# ============================================================================
# 5. RAVEN MODEL
# ============================================================================


def sinusoidal_encoding(length: int, dim: int, device):
    import torch

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class RAVEN(nn.Module):
    """Regime-Aware Variable-context Expert Network.

    Tensor convention:
        x: [batch, time, channels]
        E: [batch, num_patches, embed_dim]
    """

    def __init__(self, num_channels: int, cfg: Config):
        import torch.nn as nn

        super().__init__()
        self.num_channels = num_channels
        self.patch_len = cfg.patch_len
        self.num_patches = cfg.effective_lookback // cfg.patch_len
        self.embed_dim = cfg.embed_dim
        self.num_experts = cfg.num_experts
        self.thresholds = tuple(cfg.thresholds)
        self.dropout = cfg.dropout

        # Channel-independent patch embedding: the same linear projection is
        # applied to every feature channel, followed by channel pooling.
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.embed_dim)
        self.patch_norm = nn.LayerNorm(cfg.embed_dim)
        self.importance_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim // 2, 1),
        )

        def make_encoder(num_layers: int):
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.embed_dim * cfg.ff_multiplier,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.experts = nn.ModuleList([make_encoder(cfg.expert_layers) for _ in range(cfg.num_experts)])
        self.global_encoder = make_encoder(cfg.global_layers)
        self.alpha_head = nn.Linear(cfg.embed_dim, 1)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 1),
        )

        # Learnable non-negative redundancy penalty lambda.
        self.raw_lambda = nn.Parameter(torch.tensor(0.0))

    def _patchify(self, x):
        import torch

        batch, length, channels = x.shape
        usable = self.num_patches * self.patch_len
        x = x[:, -usable:, :]

        # [B, L, D] -> [B, D, N, patch_len]
        patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        # Reverse the patch order: most recent patch becomes index 0.
        patches = torch.flip(patches, dims=[2])
        # Shared channel-independent projection, then average channel views.
        embedded = self.patch_embed(patches)
        embedded = self.patch_norm(embedded)
        embedded = embedded.mean(dim=1)
        embedded = embedded + sinusoidal_encoding(self.num_patches, self.embed_dim, x.device).unsqueeze(0)
        return embedded

    def _instance_normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + 1e-5)

    def _run_expert_grouped(self, E, lengths, encoder):
        import torch

        batch = E.shape[0]
        z = torch.zeros(batch, self.embed_dim, device=E.device, dtype=E.dtype)
        for length_tensor in torch.unique(lengths, sorted=True):
            length = int(length_tensor.item())
            indices = torch.nonzero(lengths == length, as_tuple=False).flatten()
            if len(indices) == 0:
                continue
            h = encoder(E[indices, :length, :])
            z[indices] = h.mean(dim=1)
        return z

    def forward(self, x):
        import torch
        import torch.nn.functional as F

        x = self._instance_normalize(x)
        E = self._patchify(x)
        raw_scores = self.importance_mlp(E).squeeze(-1)
        patch_prob = F.softmax(raw_scores, dim=1)
        cumulative = torch.cumsum(patch_prob, dim=1)

        # For each threshold, choose the smallest contiguous recent prefix
        # whose cumulative importance reaches that threshold. This is the
        # practical threshold-crossing implementation of CIT.
        lengths_list = []
        for threshold in self.thresholds:
            lengths = (cumulative < threshold).sum(dim=1) + 1
            lengths = lengths.clamp(min=1, max=self.num_patches)
            lengths_list.append(lengths)

        expert_vectors = []
        for k, lengths in enumerate(lengths_list):
            expert_vectors.append(self._run_expert_grouped(E, lengths, self.experts[k]))
        z_stack = torch.stack(expert_vectors, dim=1)  # [B, K, d]

        # Raw confidence alpha_k followed by correlation-aware redundancy decay.
        alpha = F.softplus(self.alpha_head(z_stack).squeeze(-1)) + 1e-6
        normed = F.normalize(z_stack, dim=-1)
        cosine = torch.bmm(normed, normed.transpose(1, 2))
        eye = torch.eye(self.num_experts, device=x.device, dtype=x.dtype).unsqueeze(0)
        redundancy = (F.relu(cosine) * (1.0 - eye)).sum(dim=-1)
        lambda_red = F.softplus(self.raw_lambda)
        local_weights = F.softmax(torch.log(alpha) - lambda_red * redundancy, dim=1)
        z_local = (local_weights.unsqueeze(-1) * z_stack).sum(dim=1)

        # Global compressed representation branch.
        global_hidden = self.global_encoder(E)
        z_global = global_hidden.mean(dim=1)
        prediction = self.output_head(torch.cat([z_local, z_global], dim=-1)).squeeze(-1)

        aux = {
            "patch_prob": patch_prob,
            "cumulative": cumulative,
            "lengths": torch.stack(lengths_list, dim=1),
            "expert_vectors": z_stack,
            "expert_cosine": cosine,
            "expert_weights": local_weights,
            "lambda_red": lambda_red,
            "z_local": z_local,
            "z_global": z_global,
        }
        return prediction, aux


def raven_loss(prediction, target, aux, cfg: Config):
    import torch
    import torch.nn.functional as F

    mse = F.mse_loss(prediction, target)
    p = aux["patch_prob"].clamp_min(1e-8)
    entropy_penalty = (p * torch.log(p)).sum(dim=1).mean()
    cosine = aux["expert_cosine"]
    k = cosine.shape[-1]
    eye = torch.eye(k, device=cosine.device, dtype=cosine.dtype).unsqueeze(0)
    diversity_penalty = ((cosine - eye) ** 2).mean()
    total = (
        mse
        + cfg.lambda_entropy * entropy_penalty
        + cfg.lambda_diversity * diversity_penalty
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "entropy": float(entropy_penalty.detach().cpu()),
        "diversity": float(diversity_penalty.detach().cpu()),
    }


# ============================================================================
# 6. TRAINING AND EVALUATION
# ============================================================================


def make_loader(dataset: RavenSequenceDataset, cfg: Config, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_raven,
        drop_last=False,
    )


def run_epoch(model, loader, optimizer, device: str, cfg: Config, train: bool) -> Dict[str, float]:
    import torch

    model.train(train)
    totals = {"loss": 0.0, "mse": 0.0, "entropy": 0.0, "diversity": 0.0}
    count = 0
    for x, y, _, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            prediction, aux = model(x)
            loss, parts = raven_loss(prediction, y, aux, cfg)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
        n = len(y)
        count += n
        for key in totals:
            totals[key] += parts[key] * n
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


def predict(model, loader, device: str, target_mean: float, target_std: float) -> pd.DataFrame:
    import torch

    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for x, y_norm, codes, dates, y_raw in loader:
            x = x.to(device, non_blocking=True)
            pred_norm, _ = model(x)
            pred_norm = pred_norm.detach().cpu().numpy()
            true_norm = y_norm.numpy()
            pred_raw = pred_norm * target_std + target_mean
            true_raw = y_raw.numpy()
            for i in range(len(codes)):
                rows.append(
                    {
                        "ts_code": codes[i],
                        "trade_date": dates[i],
                        "pred_norm": float(pred_norm[i]),
                        "true_norm": float(true_norm[i]),
                        "pred_return": float(pred_raw[i]),
                        "true_return": float(true_raw[i]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> Dict[str, float]:
    df = predictions.dropna(subset=["pred_norm", "true_norm", "pred_return", "true_return"]).copy()
    if df.empty:
        return {}

    metrics: Dict[str, float] = {
        "pearson_corr": safe_corr(df["pred_return"], df["true_return"], "pearson"),
        "mse_normalized": float(np.mean((df["pred_norm"] - df["true_norm"]) ** 2)),
        "mae_normalized": float(np.mean(np.abs(df["pred_norm"] - df["true_norm"]))),
    }

    daily_ics: List[float] = []
    daily_rankics: List[float] = []
    for _, group in df.groupby("trade_date"):
        if len(group) < 3:
            continue
        daily_ics.append(safe_corr(group["pred_return"], group["true_return"], "pearson"))
        daily_rankics.append(safe_corr(group["pred_return"], group["true_return"], "spearman"))
    daily_ics = [x for x in daily_ics if np.isfinite(x)]
    daily_rankics = [x for x in daily_rankics if np.isfinite(x)]
    if daily_ics:
        metrics["mean_daily_ic"] = float(np.mean(daily_ics))
        metrics["icir"] = float(np.mean(daily_ics) / (np.std(daily_ics, ddof=1) + 1e-12))
        metrics["positive_ic_ratio"] = float(np.mean(np.asarray(daily_ics) > 0))
    if daily_rankics:
        metrics["mean_rankic"] = float(np.mean(daily_rankics))
        metrics["rankicir"] = float(np.mean(daily_rankics) / (np.std(daily_rankics, ddof=1) + 1e-12))
        metrics["positive_rankic_ratio"] = float(np.mean(np.asarray(daily_rankics) > 0))
    return metrics


def validation_selection_score(
    cfg: Config,
    valid_stats: Dict[str, float],
    valid_metrics: Dict[str, float],
) -> Tuple[float, str]:
    """Return a score where larger is better and the metric actually used."""
    if cfg.selection_metric == "valid_rankic":
        rankic = valid_metrics.get("mean_rankic", float("nan"))
        if np.isfinite(rankic):
            return float(rankic), "valid_rankic"
        warnings.warn("验证集 RankIC 不可用，当前轮回退为按验证损失选择模型。")
    return -float(valid_stats["loss"]), "valid_loss"


def simple_topk_backtest(predictions: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """A transparent top-K backtest using the paper's 10-day rebalance idea.

    This is intentionally simpler than Qlib's full simulator. It uses the
    realized H-day target return, equal weights the top-K names, charges a
    configurable turnover cost, and reports a cumulative equity curve.
    """

    df = predictions.copy()
    dates = sorted(pd.to_datetime(df["trade_date"]).unique())
    rebalance_dates = dates[:: max(1, cfg.rebalance_every)]
    prev_holdings: set[str] = set()
    rows: List[Dict[str, Any]] = []

    for date in rebalance_dates:
        day = df[pd.to_datetime(df["trade_date"]) == date].dropna(subset=["pred_return", "true_return"])
        if day.empty:
            continue
        selected = day.sort_values("pred_return", ascending=False).head(cfg.top_k)
        holdings = set(selected["ts_code"].astype(str))
        turnover = 1.0 if not prev_holdings else 1.0 - len(holdings & prev_holdings) / max(len(holdings), 1)
        gross_return = float(selected["true_return"].mean())
        cost = turnover * cfg.transaction_cost_bps / 10000.0
        net_return = gross_return - cost
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "n_selected": len(selected),
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net_return,
                "selected": ",".join(sorted(holdings)),
            }
        )
        prev_holdings = holdings

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve, {}
    curve["equity"] = (1.0 + curve["net_return"]).cumprod()
    total_return = float(curve["equity"].iloc[-1] - 1.0)
    years = max((curve["trade_date"].iloc[-1] - curve["trade_date"].iloc[0]).days / 365.25, 1 / 365.25)
    annualized = float((1.0 + total_return) ** (1.0 / years) - 1.0) if 1 + total_return > 0 else float("nan")
    period_std = float(curve["net_return"].std(ddof=1)) if len(curve) > 1 else float("nan")
    sharpe = float(curve["net_return"].mean() / (period_std + 1e-12) * math.sqrt(252 / cfg.rebalance_every)) if np.isfinite(period_std) else float("nan")
    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1.0
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized,
        "mean_period_return": float(curve["net_return"].mean()),
        "period_sharpe_approx": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(curve["turnover"].mean()),
        "n_rebalances": int(len(curve)),
    }
    return curve, metrics


def train_and_evaluate(
    cfg: Config,
    factors: pd.DataFrame,
    factor_cols: List[str],
) -> Dict[str, Any]:
    import torch

    set_seed(cfg.seed)
    device = choose_device(cfg)
    print(f"[train] device={device}, effective_lookback={cfg.effective_lookback}, patches={cfg.effective_lookback // cfg.patch_len}")

    blocks = build_blocks(factors, factor_cols)
    train_idx, valid_idx, test_idx, target_mean, target_std = split_indices(blocks, cfg)
    split_summary = {
        "train": summarize_split(blocks, train_idx),
        "valid": summarize_split(blocks, valid_idx),
        "test": summarize_split(blocks, test_idx),
    }
    print(f"[split detail] {json.dumps(split_summary, ensure_ascii=False)}")
    train_ds = RavenSequenceDataset(blocks, train_idx, cfg.effective_lookback, target_mean, target_std)
    valid_ds = RavenSequenceDataset(blocks, valid_idx, cfg.effective_lookback, target_mean, target_std)
    test_ds = RavenSequenceDataset(blocks, test_idx, cfg.effective_lookback, target_mean, target_std)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    valid_loader = make_loader(valid_ds, cfg, shuffle=False)
    test_loader = make_loader(test_ds, cfg, shuffle=False)

    model = RAVEN(num_channels=len(factor_cols), cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_state = None
    best_score = -float("inf")
    best_epoch = 0
    best_metric_used = cfg.selection_metric
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        valid_stats = run_epoch(model, valid_loader, None, device, cfg, train=False)
        valid_predictions = predict(model, valid_loader, device, target_mean, target_std)
        valid_metrics = evaluate_predictions(valid_predictions)
        selection_score, metric_used = validation_selection_score(cfg, valid_stats, valid_metrics)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_stats,
            "valid": valid_stats,
            "valid_metrics": valid_metrics,
            "selection_score": selection_score,
            "selection_metric_used": metric_used,
        }
        history.append(record)
        if selection_score > best_score + cfg.early_stopping_min_delta:
            best_score = selection_score
            best_epoch = epoch
            best_metric_used = metric_used
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        print(
            f"[epoch {epoch:03d}] train_loss={train_stats['loss']:.6f} "
            f"valid_loss={valid_stats['loss']:.6f} "
            f"valid_rankic={valid_metrics.get('mean_rankic', float('nan')):.6f} "
            f"select={metric_used}:{selection_score:.6f}"
        )
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"[early stop] epoch={epoch}, best_epoch={best_epoch}, "
                f"best_{best_metric_used}={best_score:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = cfg.output_dir / "raven_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "factor_cols": factor_cols,
            "target_mean": target_mean,
            "target_std": target_std,
            "device_used": device,
            "best_epoch": best_epoch,
            "best_selection_metric": best_metric_used,
            "best_selection_score": best_score,
        },
        checkpoint,
    )
    save_json({"history": history}, cfg.output_dir / "training_history.json")

    valid_predictions = predict(model, valid_loader, device, target_mean, target_std)
    valid_predictions.to_csv(cfg.output_dir / "valid_predictions.csv", index=False)
    valid_prediction_metrics = evaluate_predictions(valid_predictions)
    test_predictions = predict(model, test_loader, device, target_mean, target_std)
    test_predictions.to_csv(cfg.output_dir / "test_predictions.csv", index=False)
    prediction_metrics = evaluate_predictions(test_predictions)
    curve, backtest_metrics = simple_topk_backtest(test_predictions, cfg)
    curve.to_csv(cfg.output_dir / "topk_backtest.csv", index=False)
    save_json(
        {
            "run_summary": {
                "best_epoch": best_epoch,
                "selection_metric": best_metric_used,
                "selection_score": best_score,
                "split_summary": split_summary,
            },
            "validation_metrics": valid_prediction_metrics,
            "prediction_metrics": prediction_metrics,
            "backtest_metrics": backtest_metrics,
        },
        cfg.output_dir / "metrics.json",
    )
    print("[validation metrics]")
    print(json.dumps(valid_prediction_metrics, ensure_ascii=False, indent=2))
    print("[test metrics]")
    print(json.dumps(prediction_metrics, ensure_ascii=False, indent=2))
    print("[backtest metrics]")
    print(json.dumps(backtest_metrics, ensure_ascii=False, indent=2))
    return {
        "model": model,
        "validation_metrics": valid_prediction_metrics,
        "prediction_metrics": prediction_metrics,
        "backtest_metrics": backtest_metrics,
        "history": history,
    }


# ============================================================================
# 7. COMMAND LINE ENTRYPOINT
# ============================================================================


def load_cfg_from_args() -> Config:
    parser = argparse.ArgumentParser(description="Tushare-based RAVEN reproduction")
    parser.add_argument("--mode", choices=["download", "prepare", "train", "all"], default="all")
    parser.add_argument("--max-stocks", type=int, default=None, help="调试时限制股票数量，例如 20")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖默认 batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖默认学习率")
    parser.add_argument("--device", type=str, default=None, help="auto/cpu/cuda")
    parser.add_argument(
        "--selection-metric",
        choices=["valid_loss", "valid_rankic"],
        default=None,
        help="保存最佳模型的验证指标",
    )
    parser.add_argument("--patience", type=int, default=None, help="早停耐心轮数")
    parser.add_argument("--start-date", type=str, default=None, help="下载起始日 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="下载结束日 YYYYMMDD")
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--valid-start", type=str, default=None)
    parser.add_argument("--valid-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument(
        "--disable-cross-sectional-normalization",
        action="store_true",
        help="关闭按日去极值和截面标准化，用于消融比较",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild-features", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.max_stocks is not None:
        cfg.max_stocks = args.max_stocks
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.device is not None:
        cfg.device = args.device
    if args.selection_metric is not None:
        cfg.selection_metric = args.selection_metric
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    for field_name in [
        "start_date",
        "end_date",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    ]:
        value = getattr(args, field_name)
        if value is not None:
            setattr(cfg, field_name, value)
    if args.disable_cross_sectional_normalization:
        cfg.cross_sectional_normalize = False
    if args.force_download:
        cfg.force_download = True
    if args.force_rebuild_features:
        cfg.force_rebuild_features = True
    return cfg, args.mode


def main() -> None:
    cfg, mode = load_cfg_from_args()
    cfg.validate()
    ensure_dirs(cfg)
    save_json(asdict(cfg), cfg.root / "config_snapshot.json")

    if mode == "download":
        download_market_data(cfg)
        return

    factors, factor_cols = prepare_features(cfg)
    if mode == "prepare":
        print(f"[prepare done] {len(factors):,} rows, {len(factor_cols)} factor columns")
        return

    if mode in {"train", "all"}:
        train_and_evaluate(cfg, factors, factor_cols)


if __name__ == "__main__":
    main()
