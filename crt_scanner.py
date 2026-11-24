import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. إعدادات الإشعارات
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

SYMBOLS = [
    # القائمة الأساسية
    "ZS",
    "NVDA",
    "TSLA",
    "AAPL",
    "COST",
    "QCOM",
    "ADBE",
    "DNG",
    "GNG",
    # الإضافات الجديدة
    "AMZN",
    "CRWD",
    "COIN",
    "MSTR",
    "MU",
    "AMD",
    "SPCX",
    "META",
    "MSFT",
    "AVGO",
    "IBM",
    "BA",
    "JNJ",
    "PLTR",
    "PANW",
]

TIMEFRAME_CONFIG = {
    "1mo": {
        "label": "الشهر",
        "yf_interval": "1mo",
        "yf_period": "2y",
        "resample_4h": False,
        "lookback_bars": 4,
        "allow_unclosed": True,
    },
    "1wk": {
        "label": "الأسبوع",
        "yf_interval": "1wk",
        "yf_period": "6mo",
        "resample_4h": False,
        "lookback_bars": 8,
        "allow_unclosed": False,
    },
    "4h": {
        "label": "4 ساعات",
        "yf_interval": "1h",
        "yf_period": "60d",
        "resample_4h": True,
        "lookback_bars": 80,
        "allow_unclosed": False,
    },
}


# ==========================================
# 2. جلب البيانات وحساب المؤشرات
# ==========================================
def fetch_ohlcv(symbol: str, cfg: dict) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=cfg["yf_period"],
        interval=cfg["yf_interval"],
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise ValueError(f"لا توجد بيانات متاحة لـ {symbol} ({cfg['label']})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)

    if cfg.get("resample_4h"):
        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df = df[["open", "high", "low", "close", "volume"]].resample("4h").agg(agg_map).dropna()

    df = df.reset_index()
    time_col = df.columns[0]
    df = df.rename(columns={time_col: "datetime"})

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ==========================================
# 3. الدايفرجنس السلبي، نسبة الذيل، والأهداف
# ==========================================
def has_bearish_divergence(price: pd.Series, osc: pd.Series, start_idx: int, end_idx: int) -> bool:
    if start_idx < 1 or end_idx <= start_idx or end_idx >= len(price):
        return False

    p_win = price.iloc[start_idx : end_idx + 1]
    if len(p_win) < 3:
        return False

    mid = len(p_win) // 2
    first_half_max_idx = p_win.iloc[:mid].idxmax()
    second_half_max_idx = p_win.iloc[mid:].idxmax()

    if first_half_max_idx >= second_half_max_idx:
        return False

    return (price.loc[second_half_max_idx] > price.loc[first_half_max_idx]) and (
        osc.loc[second_half_max_idx] < osc.loc[first_half_max_idx]
    )


def calc_upper_wick_ratio(row: pd.Series) -> float:
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return 0.0
    upper_wick = row["high"] - max(row["open"], row["close"])
    return max(0.0, (upper_wick / candle_range) * 100.0)


def format_simple_price(val: float) -> str:
    if abs(val) >= 100:
        return f"{round(val)}"
    elif abs(val) >= 10:
        return f"{val:.1f}".rstrip("0").rstrip(".")
    else:
        return f"{val:.2f}".rstrip("0").rstrip(".")


def get_remaining_targets(df: pd.DataFrame, sweep_idx: int, current_price: float) -> list[str]:
    history = df.iloc[:sweep_idx]
    if history.empty:
        return []

    candidate_lows = sorted(
        [float(x) for x in history["low"] if float(x) < current_price],
        reverse=True,
    )

    targets = []
    min_gap = max(current_price * 0.015, 0.5)

    for lvl in candidate_lows:
        if all(abs(lvl - prev) >= min_gap for prev in [float(t) for t in targets]):
            targets.append(format_simple_price(lvl))
        if len(targets) >= 3:
            break

    return targets


# ==========================================
# 4. اكتشاف نموذج CRT
# ==========================================
def detect_crt_signals(symbol: str, interval: str, cfg: dict) -> list[str]:
    df = fetch_ohlcv(symbol, cfg)
    if len(df) < 6:
        return []

    df["rsi"] = compute_rsi(df["close"], period=14)
    df["obv"] = compute_obv(df["close"], df["volume"])

    last_valid_idx = len(df) - 1 if cfg["allow_unclosed"] else len(df) - 2
    if last_valid_idx < 3:
        return []

    current_price = float(df.loc[len(df) - 1, "close"])
    start_search_idx = max(2, last_valid_idx - cfg["lookback_bars"] + 1)

    alerts = []

    for idx in range(start_search_idx, last_valid_idx + 1):
        ref_bar = df.iloc[idx - 1]
        crt_bar = df.iloc[idx]

        range_high = float(ref_bar["high"])
        range_low = float(ref_bar["low"])
        crt_high = float(crt_bar["high"])
        crt_close = float(crt_bar["close"])

        swept_high = crt_high > range_high
        closed_inside_range = range_low <= crt_close <= range_high

        if not (swept_high and closed_inside_range):
            continue

        div_start = max(1, idx - 8)
        obv_bear_div = has_bearish_divergence(df["high"], df["obv"], div_start, idx)
        rsi_bear_div = has_bearish_divergence(df["high"], df["rsi"], div_start, idx)

        active_divs = []
        if obv_bear_div:
            active_divs.append("OBV")
        if rsi_bear_div:
            active_divs.append("RSI")

        if not active_divs:
            continue

        wick_pct = calc_upper_wick_ratio(crt_bar)
        targets = get_remaining_targets(df, idx, current_price)

        div_text = " + ".join(active_divs)
        targets_text = " ، ".join(targets) if targets else "غير متوفرة حالياً"
        status_note = " (شمعة جارية)" if (idx == len(df) - 1 and cfg["allow_unclosed"]) else ""

        msg = (
            f"📊 الشركة: {symbol}{status_note}\n"
            f"🕒 الفريم: {cfg['label']}\n"
            f"🎯 الأهداف: {targets_text}\n"
            f"📏 نسبة الذيل: {wick_pct:.0f}%\n"
            f"📉 دايفرجنس سلبي: {div_text}"
        )
        alerts.append(msg)

    return alerts


# ==========================================
# 5. إرسال التنبيهات إلى تيليجرام
# ==========================================
def send_telegram_message(text: str) -> None:
    if "YOUR_" in TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_CHAT_ID:
        print(text)
        print("-" * 35)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def run_scanner() -> None:
    for symbol in SYMBOLS:
        for interval, cfg in TIMEFRAME_CONFIG.items():
            try:
                signals = detect_crt_signals(symbol, interval, cfg)
                for sig in signals:
                    send_telegram_message(sig)
                    time.sleep(0.3)
            except Exception as e:
                print(f"[{symbol} | {interval}] خطأ أثناء الفحص: {e}")


if __name__ == "__main__":
    run_scanner()
