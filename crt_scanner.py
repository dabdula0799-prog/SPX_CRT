import os
import time
import smtplib
from email.message import EmailMessage
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. إعدادات الإشعارات وقائمة الأسهم
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

SMTP_SERVER = (os.getenv("SMTP_SERVER") or "smtp.gmail.com").strip()
_raw_port = (os.getenv("SMTP_PORT") or "465").strip()
SMTP_PORT = int(_raw_port) if _raw_port.isdigit() else 465

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "YOUR_EMAIL@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "YOUR_APP_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "RECEIVER_EMAIL@gmail.com")

SYMBOLS = [
    "ZS",
    "NVDA",
    "TSLA",
    "AAPL",
    "COST",
    "QCOM",
    "ADBE",
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
# 2. جلب البيانات مع إعادة المحاولة وتهدئة الطلبات
# ==========================================
def fetch_ohlcv(symbol: str, cfg: dict, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """جلب بيانات OHLCV من Yahoo Finance مع إعادة المحاولة عند انقطاع الاتصال."""
    last_err = None
    df = None

    for attempt in range(1, retries + 1):
        try:
            time.sleep(delay * attempt)
            df = yf.download(
                tickers=symbol,
                period=cfg["yf_period"],
                interval=cfg["yf_interval"],
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df is not None and not df.empty:
                break
            last_err = ValueError(f"لا توجد بيانات متاحة لـ {symbol} ({cfg['label']})")
        except Exception as e:
            last_err = e

    if df is None or df.empty:
        raise last_err or ValueError(f"لا توجد بيانات متاحة لـ {symbol} ({cfg['label']})")

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
# 3. الدايفرجنس (سلبي وإيجابي)، نسبة الذيل، والأهداف
# ==========================================
def has_bearish_divergence(price: pd.Series, osc: pd.Series, start_idx: int, end_idx: int) -> bool:
    """دايفرجنس سلبي: قمة سعرية أعلى مقابل قمة أضعف على المؤشر."""
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


def has_bullish_divergence(price: pd.Series, osc: pd.Series, start_idx: int, end_idx: int) -> bool:
    """دايفرجنس إيجابي: قاع سعري أدنى مقابل قاع أعلى/أقوى على المؤشر."""
    if start_idx < 1 or end_idx <= start_idx or end_idx >= len(price):
        return False

    p_win = price.iloc[start_idx : end_idx + 1]
    if len(p_win) < 3:
        return False

    mid = len(p_win) // 2
    first_half_min_idx = p_win.iloc[:mid].idxmin()
    second_half_min_idx = p_win.iloc[mid:].idxmin()

    if first_half_min_idx >= second_half_min_idx:
        return False

    return (price.loc[second_half_min_idx] < price.loc[first_half_min_idx]) and (
        osc.loc[second_half_min_idx] > osc.loc[first_half_min_idx]
    )


def calc_wick_ratio(row: pd.Series, direction: str) -> float:
    """حساب نسبة الذيل العلوي للسلبي أو السفلي للإيجابي مقارنة بالمدى الكلي للشمعة."""
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return 0.0
    if direction == "bearish":
        wick = row["high"] - max(row["open"], row["close"])
    else:
        wick = min(row["open"], row["close"]) - row["low"]
    return max(0.0, (wick / candle_range) * 100.0)


def format_simple_price(val: float) -> str:
    if abs(val) >= 100:
        return f"{round(val)}"
    elif abs(val) >= 10:
        return f"{val:.1f}".rstrip("0").rstrip(".")
    else:
        return f"{val:.2f}".rstrip("0").rstrip(".")


def get_remaining_targets(
    df: pd.DataFrame, sweep_idx: int, current_price: float, direction: str
) -> list[str]:
    """استخراج حتى 3 أهداف سعرية متبقية (للأسفل للسلبي، وللأعلى للإيجابي) وتبسيط أرقامها."""
    history = df.iloc[:sweep_idx]
    if history.empty:
        return []

    min_gap = max(current_price * 0.015, 0.5)
    targets = []

    if direction == "bearish":
        candidates = sorted(
            [float(x) for x in history["low"] if float(x) < current_price],
            reverse=True,
        )
    else:
        candidates = sorted(
            [float(x) for x in history["high"] if float(x) > current_price],
            reverse=False,
        )

    for lvl in candidates:
        if all(abs(lvl - prev) >= min_gap for prev in [float(t) for t in targets]):
            targets.append(format_simple_price(lvl))
        if len(targets) >= 3:
            break

    return targets


# ==========================================
# 4. اكتشاف نموذج CRT
# ==========================================
def detect_crt_signals(symbol: str, interval: str, cfg: dict) -> list[tuple[str, str]]:
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
        crt_low = float(crt_bar["low"])
        crt_close = float(crt_bar["close"])

        closed_inside_range = range_low <= crt_close <= range_high
        if not closed_inside_range:
            continue

        div_start = max(1, idx - 8)
        status_note = " (شمعة جارية)" if (idx == len(df) - 1 and cfg["allow_unclosed"]) else ""

        # قراءة الدايفرجنس السلبي عند سحب القمة والإغلاق داخل النطاق
        if crt_high > range_high:
            obv_bear = has_bearish_divergence(df["high"], df["obv"], div_start, idx)
            rsi_bear = has_bearish_divergence(df["high"], df["rsi"], div_start, idx)
            bear_divs = [d for d, ok in [("OBV", obv_bear), ("RSI", rsi_bear)] if ok]

            if bear_divs:
                wick_pct = calc_wick_ratio(crt_bar, "bearish")
                targets = get_remaining_targets(df, idx, current_price, "bearish")
                div_text = " + ".join(bear_divs)
                targets_text = " ، ".join(targets) if targets else "غير متوفرة حالياً"

                subject = f"إشارة CRT سلبية | {symbol} ({cfg['label']}) - {div_text}"
                msg = (
                    f"📊 الشركة: {symbol}{status_note}\n"
                    f"🕒 الفريم: {cfg['label']}\n"
                    f"🎯 الأهداف: {targets_text}\n"
                    f"📏 نسبة الذيل العلوي: {wick_pct:.0f}%\n"
                    f"📉 دايفرجنس سلبي: {div_text}"
                )
                alerts.append((subject, msg))

        # قراءة الدايفرجنس الإيجابي عند كسر القاع والإغلاق داخل النطاق
        if crt_low < range_low:
            obv_bull = has_bullish_divergence(df["low"], df["obv"], div_start, idx)
            rsi_bull = has_bullish_divergence(df["low"], df["rsi"], div_start, idx)
            bull_divs = [d for d, ok in [("OBV", obv_bull), ("RSI", rsi_bull)] if ok]

            if bull_divs:
                wick_pct = calc_wick_ratio(crt_bar, "bullish")
                targets = get_remaining_targets(df, idx, current_price, "bullish")
                div_text = " + ".join(bull_divs)
                targets_text = " ، ".join(targets) if targets else "غير متوفرة حالياً"

                subject = f"إشارة CRT إيجابية | {symbol} ({cfg['label']}) - {div_text}"
                msg = (
                    f"📊 الشركة: {symbol}{status_note}\n"
                    f"🕒 الفريم: {cfg['label']}\n"
                    f"🎯 الأهداف: {targets_text}\n"
                    f"📏 نسبة الذيل السفلي: {wick_pct:.0f}%\n"
                    f"📈 دايفرجنس إيجابي: {div_text}"
                )
                alerts.append((subject, msg))

    return alerts


# ==========================================
# 5. إرسال التنبيهات (تيليجرام + البريد الإلكتروني)
# ==========================================
def send_telegram_message(text: str) -> None:
    if "YOUR_" in TELEGRAM_BOT_TOKEN or "YOUR_" in TELEGRAM_CHAT_ID:
        print("[Telegram Preview]\n" + text)
        print("-" * 35)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()


def send_email_alert(subject: str, body: str) -> None:
    if "YOUR_" in EMAIL_SENDER or "YOUR_" in EMAIL_PASSWORD or "RECEIVER_" in EMAIL_RECEIVER:
        print(f"[Email Preview] Subject: {subject}")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)


def run_scanner() -> None:
    for symbol in SYMBOLS:
        for interval, cfg in TIMEFRAME_CONFIG.items():
            try:
                signals = detect_crt_signals(symbol, interval, cfg)
                for subject, body in signals:
                    send_telegram_message(body)
                    send_email_alert(subject, body)
                    time.sleep(0.3)
            except Exception as e:
                print(f"[{symbol} | {interval}] خطأ أثناء الفحص: {e}")


if __name__ == "__main__":
    run_scanner()
