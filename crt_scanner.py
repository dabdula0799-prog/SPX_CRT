import os
import time
import requests
import pandas as pd
import numpy as np

# ==========================================
# 1. إعدادات الإشعارات والـ API
# ==========================================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "YOUR_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# قائمة الشركات المطلوب فحصها (أضف أو عدّل الرموز حسب حاجتك)
SYMBOLS = [
    "LLY",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "CRWD",
    "QCOM",
]

# إعدادات الفريمات ومدى البحث (Lookback):
# - فريم الشهر (1month): مدى أوسع (مثلاً 12 شمعة شهرية) ولا نمنع التنبيه حتى لو كان سحب السيولة قبل شهر أو شهرين أو أكثر، وحتى لو لم تُغلق الشمعة الحالية بعد.
# - فريم الأسبوع (1week) وفريم 4 ساعات (4h): من شهر إلى شهرين تقريباً كمدى بحث مناسب.
TIMEFRAME_CONFIG = {
    "1month": {
        "label": "الشهر",
        "outputsize": 14,
        "lookback_bars": 4,       # يسمح برصد سحب السيولة الأقدم (شهر، شهرين، أو أكثر)
        "allow_unclosed": True,   # يرسل التنبيه فوراً حتى لو بقي أيام على إغلاق الشمعة الشهرية
    },
    "1week": {
        "label": "الأسبوع",
        "outputsize": 16,
        "lookback_bars": 8,       # يغطي تقريباً شهر إلى شهرين على الفريم الأسبوعي
        "allow_unclosed": False,
    },
    "4h": {
        "label": "4 ساعات",
        "outputsize": 160,
        "lookback_bars": 80,      # يغطي تقريباً شهر إلى شهرين على فريم 4 ساعات
        "allow_unclosed": False,
    },
}


# ==========================================
# 2. جلب البيانات وحساب المؤشرات
# ==========================================
def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 100) -> pd.DataFrame:
    """جلب بيانات السعر والحجم من Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"خطأ في بيانات {symbol} ({interval}): {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """حساب مؤشر OBV."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


# ==========================================
# 3. فحص الدايفرجنس السلبي، نسبة الذيل، والأهداف
# ==========================================
def has_bearish_divergence(price: pd.Series, osc: pd.Series, start_idx: int, end_idx: int) -> bool:
    """
    يتحقق من وجود دايفرجنس سلبي بين قمتين ضمن المجال المحدد:
    السعر يحقق قمة أعلى، بينما المؤشر يحقق قمة أدنى أو أضعف.
    """
    if start_idx < 1 or end_idx <= start_idx or end_idx >= len(price):
        return False

    p_win = price.iloc[start_idx : end_idx + 1]
    o_win = osc.iloc[start_idx : end_idx + 1]

    if len(p_win) < 3:
        return False

    # تحديد قمتين داخل النافذة: قمة سابقة وقمة لاحقة
    mid = len(p_win) // 2
    first_half_max_idx = p_win.iloc[:mid].idxmax()
    second_half_max_idx = p_win.iloc[mid:].idxmax()

    if first_half_max_idx >= second_half_max_idx:
        return False

    price_1 = price.loc[first_half_max_idx]
    price_2 = price.loc[second_half_max_idx]

    osc_1 = osc.loc[first_half_max_idx]
    osc_2 = osc.loc[second_half_max_idx]

    return (price_2 > price_1) and (osc_2 < osc_1)


def calc_upper_wick_ratio(row: pd.Series) -> float:
    """
    حساب نسبة الذيل العلوي مقارنة بالمدى الكلي للشمعة (High - Low).
    """
    candle_range = row["high"] - row["low"]
    if candle_range <= 0:
        return 0.0
    upper_wick = row["high"] - max(row["open"], row["close"])
    return max(0.0, (upper_wick / candle_range) * 100.0)


def format_simple_price(val: float) -> str:
    """
    تنسيق الأرقام بشكل مبسط ومباشر دون تعقيد أو أرقام عشرية كثيرة.
    """
    if abs(val) >= 100:
        return f"{round(val)}"
    elif abs(val) >= 10:
        return f"{val:.1f}".rstrip("0").rstrip(".")
    else:
        return f"{val:.2f}".rstrip("0").rstrip(".")


def get_remaining_targets(df: pd.DataFrame, sweep_idx: int, current_price: float) -> list[str]:
    """
    استخراج الأهداف السعرية المتبقية للأسفل (قيعان/مستويات سابقة تحت السعر الحالي)
    وتبسيط أرقامها.
    """
    history = df.iloc[:sweep_idx]
    if history.empty:
        return []

    # جمع قيعان سابقة تقع أسفل السعر الحالي
    candidate_lows = sorted(
        [float(x) for x in history["low"] if float(x) < current_price],
        reverse=True,
    )

    targets = []
    min_gap = max(current_price * 0.015, 0.5)  # تجنب تكرار مستويات متقاربة جداً

    for lvl in candidate_lows:
        if all(abs(lvl - prev) >= min_gap for prev in [float(t) for t in targets]):
            targets.append(format_simple_price(lvl))
        if len(targets) >= 3:
            break

    return targets


# ==========================================
# 4. منطق اكتشاف نموذج CRT على الفريم
# ==========================================
def detect_crt_signals(symbol: str, interval: str, cfg: dict) -> list[str]:
    df = fetch_ohlcv(symbol, interval, outputsize=cfg["outputsize"])
    if len(df) < 6:
        return []

    df["rsi"] = compute_rsi(df["close"], period=14)
    df["obv"] = compute_obv(df["close"], df["volume"])

    # إذا لم يكن مسموحاً بالشمعة غير المغلقة، نستبعد آخر شمعة جارية
    last_valid_idx = len(df) - 1 if cfg["allow_unclosed"] else len(df) - 2
    if last_valid_idx < 3:
        return []

    current_price = float(df.loc[len(df) - 1, "close"])
    start_search_idx = max(2, last_valid_idx - cfg["lookback_bars"] + 1)

    alerts = []

    for idx in range(start_search_idx, last_valid_idx + 1):
        # شمعة النطاق المرجعية (الشمعة السابقة التي تحدد High و Low النطاق)
        ref_bar = df.iloc[idx - 1]
        # شمعة CRT (الشمعة التي تسحب السيولة وتقفل داخل النطاق)
        crt_bar = df.iloc[idx]

        range_high = float(ref_bar["high"])
        range_low = float(ref_bar["low"])

        crt_high = float(crt_bar["high"])
        crt_close = float(crt_bar["close"])

        # 1) سحب السيولة من الأعلى + إغلاق شمعة CRT داخل النطاق
        swept_high = crt_high > range_high
        closed_inside_range = (range_low <= crt_close <= range_high)

        if not (swept_high and closed_inside_range):
            continue

        # 2) فحص الدايفرجنس السلبي على OBV و RSI (يكفي أحدهما أو كلاهما معاً)
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

        # 3) حساب نسبة الذيل والأهداف المتبقية
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
