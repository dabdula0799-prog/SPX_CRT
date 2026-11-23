import os
import time
import logging
from datetime import datetime, time as dt_time
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# استيراد دوال الاستراتيجية من الملف الثاني (الكود الثاني)
from strategy import analyze_market, generate_chart_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# إعدادات Bybit و التنبيهات (يفضل وضعها في متغيرات البيئة .env)
BYBIT_BASE_URL = "https://api.bybit.com"
SYMBOL = os.getenv("BYBIT_SYMBOL", "SP500")  # تحقق من الرمز المعتمد في حسابك/منتجك على Bybit
TIMEFRAME = os.getenv("BYBIT_TIMEFRAME", "30")  # بالدقائق، مثلاً 30 للـ 30 دقيقة

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your_email@example.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "YOUR_APP_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "target_email@example.com")

# نافذة العمل: من 02:00 صباحاً حتى 00:30 بعد منتصف الليل
START_TIME = dt_time(2, 0)
END_TIME = dt_time(0, 30)


def is_within_active_window(now: datetime | None = None) -> bool:
    """يتحقق هل الوقت الحالي ضمن الفترة من 02:00 صباحاً إلى 00:30 منتصف الليل."""
    current = (now or datetime.now()).time()
    # بما أن الفترة تعبر منتصف الليل (02:00 -> 00:30)
    return current >= START_TIME or current <= END_TIME


def fetch_bybit_klines(symbol: str = SYMBOL, interval: str = TIMEFRAME, limit: int = 200) -> list[dict]:
    """جلب الشموع (Klines) من Bybit."""
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

    # Bybit تعيد البيانات مرتبة من الأحدث للأقدم عادةً، لذا نعكسها لتكون تصاعدية زمنياً
    raw_list = list(reversed(data["result"]["list"]))
    candles = []
    for row in raw_list:
        candles.append({
            "timestamp": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return candles


def send_telegram_message(text: str, image_path: str | None = None) -> None:
    """إرسال رسالة نصية أو صورة عبر تيليجرام."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("بيانات تيليجرام غير مكتملة.")
        return

    if image_path and os.path.exists(image_path):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
            resp = requests.post(url, data=payload, files=files, timeout=20)
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, json=payload, timeout=15)

    resp.raise_for_status()
    logging.info("تم إرسال التنبيه إلى تيليجرام.")


def send_email_alert(subject: str, body: str) -> None:
    """إرسال تنبيه عبر البريد الإلكتروني."""
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        logging.warning("بيانات البريد الإلكتروني غير مكتملة.")
        return

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    logging.info("تم إرسال التنبيه إلى البريد الإلكتروني.")


def run_bot(poll_seconds: int = 60) -> None:
    """الحلقة الرئيسية لمراقبة السوق."""
    logging.info("بدء تشغيل البوت الرئيسي...")
    last_alert_candle_ts: int | None = None

    while True:
        try:
            if is_within_active_window():
                candles = fetch_bybit_klines()
                if candles:
                    latest_ts = candles[-1]["timestamp"]
                    signal = analyze_market(candles)

                    if signal and signal.get("triggered") and latest_ts != last_alert_candle_ts:
                        last_alert_candle_ts = latest_ts
                        message = signal.get("message", "تحققت شروط الاستراتيجية.")
                        chart_path = generate_chart_image(candles, signal)

                        send_telegram_message(message, image_path=chart_path)
                        send_email_alert(
                            subject=f"تنبيه استراتيجية {SYMBOL}",
                            body=message
                        )
            else:
                logging.debug("خارج نافذة التشغيل المحددة.")

        except Exception as exc:
            logging.error(f"حدث خطأ أثناء التشغيل: {exc}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_bot()
