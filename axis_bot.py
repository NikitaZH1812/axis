
"""
Axis Life Score — Telegram MVP
Requirements: see requirements.txt
Run:
  1) Copy .env.example to .env and put your TELEGRAM_BOT_TOKEN there.
  2) pip install -r requirements.txt
  3) python axis_bot.py
"""

import os
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)

from sweet_spots import save_log, recalculate_all_sweet_spots, get_sweet_spots

def on_all_inputs_received(user_id, mood, sleep, activity, focus, social):
    # 1) зберігаємо лог
    save_log(user_id, mood, sleep, activity, focus, social)

    # 2) оновлюємо sweet spots (коли даних ≥10 днів — побачиш логи)
    recalculate_all_sweet_spots(user_id)

    # 3) (опційно) дістаємо поточні sweet spots і використовуємо у розрахунках Stability
    spots = get_sweet_spots(user_id)
    # ...тут твоя логіка розрахунку Stability/Life Score з урахуванням spots ...



# ---------------------- Config ----------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Model hyperparams (MVP defaults)
ALPHA = 0.7   # Mood = ALPHA * M_eff + (1-ALPHA) * M_hat
BETA  = 0.65  # LifeScore = BETA * Mood + (1-BETA) * Stability

# Stability weights (sum to 1.0)
W_S = 0.35  # sleep
W_F = 0.30  # focus
W_A = 0.20  # activity
W_SC = 0.15 # social

# Lag weights (today, yesterday, 2 days ago)
LAG_WEIGHTS = [0.6, 0.3, 0.1]

# Defaults for "sweet spots" when not enough data
DEFAULTS = {
    "sleep":     {"optimal": 8.0, "tolerance": 2.0, "min": 0.0, "max": 12.0},
    "activity":  {"optimal": 5.0, "tolerance": 2.0, "min": 0.0, "max": 10.0},
    "focus":     {"optimal": 6.0, "tolerance": 2.0, "min": 0.0, "max": 10.0},
    "social":    {"optimal": 3.0, "tolerance": 2.0, "min": 0.0, "max": 10.0},
}

# --------------- Simple persistence helpers ---------------
def user_file(user_id: int) -> Path:
    return DATA_DIR / f"user_{user_id}.json"

def load_user(user_id: int) -> Dict[str, Any]:
    f = user_file(user_id)
    if f.exists():
        return json.loads(f.read_text())
    return {"entries": []}  # list of {date: "YYYY-MM-DD", mood, sleep, activity, focus, social}

def save_user(user_id: int, data: Dict[str, Any]) -> None:
    user_file(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def get_entry(entries: List[Dict[str, Any]], date_str: str) -> Dict[str, Any]:
    for e in entries:
        if e["date"] == date_str:
            return e
    e = {"date": date_str}
    entries.append(e)
    return e

def get_past_entries(entries: List[Dict[str, Any]], days_back: int) -> List[Dict[str, Any]]:
    target_dates = [(datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days_back)]
    return [e for e in entries if e["date"] in target_dates]

# --------------- Normalization & Model ---------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_from_optimal(actual: float, optimal: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    val = 1.0 - abs(actual - optimal) / tolerance
    return clamp(val, 0.0, 1.0)

def with_lag(values: List[float]) -> float:
    # values: [today, yesterday, 2days]
    total = 0.0
    weight_sum = 0.0
    for v, w in zip(values, LAG_WEIGHTS):
        total += v * w
        weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0

def derive_optimal_from_history(history: List[Dict[str, Any]], key: str, window: int = 14) -> Dict[str, float]:
    # Naive: optimal = mean of last 'window' values where mood >= user's median mood
    if not history:
        d = DEFAULTS[key]
        return {"optimal": d["optimal"], "tolerance": d["tolerance"]}
    recent = sorted(history, key=lambda e: e["date"], reverse=True)[:window]
    vals = [e.get(key) for e in recent if key in e and isinstance(e.get(key), (int, float))]
    moods = [e.get("mood") for e in recent if "mood" in e and isinstance(e.get("mood"), (int, float))]
    if len(vals) < 5 or len(moods) < 5:
        d = DEFAULTS[key]
        return {"optimal": d["optimal"], "tolerance": d["tolerance"]}
    # median mood
    smoods = sorted(moods)
    mid = len(smoods)//2
    median_mood = (smoods[mid] if len(smoods)%2==1 else (smoods[mid-1]+smoods[mid])/2)
    # take values on days with mood >= median
    good_vals = [e.get(key) for e in recent if "mood" in e and e["mood"] >= median_mood and key in e]
    if not good_vals:
        d = DEFAULTS[key]
        return {"optimal": d["optimal"], "tolerance": d["tolerance"]}
    optimal = sum(good_vals)/len(good_vals)
    # tolerance as (max(1.5, std dev)) fallback
    mean = sum(vals)/len(vals)
    variance = sum((x-mean)**2 for x in vals)/len(vals)
    std = math.sqrt(variance)
    tolerance = max(1.5, std if std>0 else 2.0)
    return {"optimal": optimal, "tolerance": tolerance}

def compute_stability(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    # gather today's and past two days' values per factor
    today = datetime.now().strftime("%Y-%m-%d")
    xkeys = ["sleep", "activity", "focus", "social"]
    # get three day values (today, yesterday, 2days)
    triple = []
    for d in range(3):
        ds = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        entry = next((e for e in history if e["date"] == ds), None)
        triple.append(entry)
    # derive optimal/tolerance per factor from last 14 days
    derived = {k: derive_optimal_from_history(history, k, window=14) for k in xkeys}
    # normalize with optimal/tolerance per day and then apply lag
    effs = {}
    for k in xkeys:
        per_day_norm = []
        for entry in triple:
            if entry is None or k not in entry:
                per_day_norm.append(0.0)  # missing data counts as 0 (can refine later)
            else:
                a = float(entry[k])
                opt = derived[k]["optimal"]
                tol = derived[k]["tolerance"]
                per_day_norm.append(normalize_from_optimal(a, opt, tol))
        effs[k] = with_lag(per_day_norm)
    stability = (
        W_S * effs["sleep"] +
        W_F * effs["focus"] +
        W_A * effs["activity"] +
        W_SC * effs["social"]
    )
    return {"value": stability, "components": effs, "optimal": derived}

def compute_m_hat(stability_value: float) -> float:
    # MVP simplification: use stability as a proxy for predicted mood
    return stability_value

def normalize_mood(mood_1_10: float) -> float:
    # map 1..10 to 0..1
    return (float(mood_1_10) - 1.0) / 9.0

def compute_life_score(entry_date: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Use latest 3 days around entry_date (assuming entry_date is today in MVP)
    stability = compute_stability(history)
    # mood effective with lag (if we have past values)
    # We'll compute lagged mood from normalized mood on the last 3 days
    moods = []
    for d in range(3):
        ds = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        entry = next((e for e in history if e["date"] == ds), None)
        if entry and "mood" in entry:
            moods.append(normalize_mood(entry["mood"]))
        else:
            moods.append(0.0)
    m_eff = with_lag(moods)
    m_hat = compute_m_hat(stability["value"])
    mood = ALPHA * m_eff + (1-ALPHA) * m_hat
    life_raw = BETA * mood + (1 - BETA) * stability["value"]
    life_score = round(100 * life_raw)
    # qualitative
    if life_score < 50:
        band = "🔴 низький"
    elif life_score < 75:
        band = "🟡 стабільний"
    else:
        band = "🟢 високий"
    return {
        "life_score": life_score,
        "band": band,
        "mood": mood,
        "m_eff": m_eff,
        "m_hat": m_hat,
        "stability": stability
    }

# --------------- Telegram Bot ---------------
ASK_MOOD, ASK_SLEEP, ASK_ACTIVITY, ASK_FOCUS, ASK_SOCIAL, CONFIRM = range(6)

def fmt_optimal_line(k: str, opt: float, tol: float) -> str:
    name = {
        "sleep": "сон",
        "activity": "активність",
        "focus": "фокус",
        "social": "соціум"
    }[k]
    return f"• {name}: оптимум ≈ {opt:.1f}, толеранс ±{tol:.1f}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Це MVP Axis.\n"
        "Натисни /log щоб зафіксувати сьогоднішні дані і отримати свій Life Score.\n"
        "Команди: /log, /score (показати сьогоднішній), /week (середнє за 7 днів), /help"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/log — ввести показники за сьогодні\n"
        "/score — обчислити і показати сьогоднішній Life Score\n"
        "/week — середній Life Score за 7 днів\n"
        "/export — отримати JSON із вашими записами"
    )

async def log_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "Оціни свій настрій (Mood) від 1 до 10:",
        reply_markup=ReplyKeyboardMarkup([ [str(i) for i in range(1,11)] ], one_time_keyboard=True, resize_keyboard=True)
    )
    return ASK_MOOD

async def ask_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        mood = int(update.message.text)
        if mood < 1 or mood > 10:
            raise ValueError()
    except:
        await update.message.reply_text("Введи число від 1 до 10.")
        return ASK_MOOD
    context.user_data["mood"] = mood
    await update.message.reply_text(
        "Скільки годин ти спав(-ла) останню ніч? (0–12)",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASK_SLEEP

async def ask_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        sleep = float(update.message.text.replace(",", "."))
        if sleep < 0 or sleep > 12:
            raise ValueError()
    except:
        await update.message.reply_text("Введи число годин сну від 0 до 12 (наприклад, 7.5).")
        return ASK_SLEEP
    context.user_data["sleep"] = sleep
    await update.message.reply_text("Оціни рівень фізичної активності (0–10):")
    return ASK_ACTIVITY

async def ask_focus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        activity = float(update.message.text.replace(",", "."))
        if activity < 0 or activity > 10:
            raise ValueError()
    except:
        await update.message.reply_text("Введи число від 0 до 10.")
        return ASK_ACTIVITY
    context.user_data["activity"] = activity
    await update.message.reply_text("Оціни фокус/продуктивність (0–10):")
    return ASK_FOCUS

async def ask_social(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        focus = float(update.message.text.replace(",", "."))
        if focus < 0 or focus > 10:
            raise ValueError()
    except:
        await update.message.reply_text("Введи число від 0 до 10.")
        return ASK_FOCUS
    context.user_data["focus"] = focus
    await update.message.reply_text("Оціни соціальну взаємодію/зв'язок (0–10):")
    return ASK_SOCIAL

async def finalize_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        social = float(update.message.text.replace(",", "."))
        if social < 0 or social > 10:
            raise ValueError()
    except:
        await update.message.reply_text("Введи число від 0 до 10.")
        return ASK_SOCIAL
    context.user_data["social"] = social

    user_id = update.effective_user.id
    data = load_user(user_id)
    entry = get_entry(data["entries"], today_str())
    entry.update({
        "mood": int(context.user_data["mood"]),
        "sleep": float(context.user_data["sleep"]),
        "activity": float(context.user_data["activity"]),
        "focus": float(context.user_data["focus"]),
        "social": float(context.user_data["social"]),
        "ts": datetime.now().isoformat(timespec="seconds")
    })
    save_user(user_id, data)

    # Compute score immediately
    res = compute_life_score(today_str(), data["entries"])

    lines = [
        f"✅ Запис збережено ({today_str()}).",
        f"Life Score: {res['life_score']} {res['band']}",
        f"Mood (ядро): {res['mood']*100:.0f} (m_eff={res['m_eff']*100:.0f}, m_hat={res['m_hat']*100:.0f})",
        f"Stability: {res['stability']['value']*100:.0f}",
        "— Внесок факторів: "
        f"сон {res['stability']['components']['sleep']*100:.0f} • "
        f"фокус {res['stability']['components']['focus']*100:.0f} • "
        f"активність {res['stability']['components']['activity']*100:.0f} • "
        f"соціум {res['stability']['components']['social']*100:.0f}",
        "— Оптимальні зони:",
    ]
    for k, v in res["stability"]["optimal"].items():
        lines.append(fmt_optimal_line(k, v["optimal"], v["tolerance"]))

    await update.message.reply_text("\n".join(lines))
    return ConversationHandler.END

async def score_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    data = load_user(user_id)
    if not data["entries"] or data["entries"][-1]["date"] != today_str():
        await update.message.reply_text("Немає сьогоднішнього запису. Натисни /log.")
        return
    res = compute_life_score(today_str(), data["entries"])
    await update.message.reply_text(
        f"Life Score: {res['life_score']} {res['band']}\n"
        f"Mood (ядро): {res['mood']*100:.0f}\n"
        f"Stability: {res['stability']['value']*100:.0f}"
    )

async def week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    data = load_user(user_id)
    if not data["entries"]:
        await update.message.reply_text("Немає даних. Почни з /log.")
        return
    # compute daily scores for last 7 days (if entries exist)
    scores = []
    for d in range(7):
        ds = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        if any(e["date"] == ds for e in data["entries"]):
            res = compute_life_score(ds, data["entries"])
            scores.append(res["life_score"])
    if not scores:
        await update.message.reply_text("Немає достатньо даних за останні 7 днів.")
        return
    avg = sum(scores)/len(scores)
    await update.message.reply_text(f"Середній Life Score за {len(scores)} днів: {avg:.1f}")

async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    data = load_user(user_id)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    # send as code block (small data) or as file if large
    if len(payload) < 3500:
        await update.message.reply_text(f"```\n{payload}\n```", parse_mode="MarkdownV2")
    else:
        p = DATA_DIR / f"export_{user_id}.json"
        p.write_text(payload)
        await update.message.reply_document(document=str(p), filename=p.name)

def build_app(token: str):
    app = ApplicationBuilder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("log", log_cmd)],
        states={
            ASK_MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_sleep)],
            ASK_SLEEP: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_activity)],
            ASK_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_focus)],
            ASK_FOCUS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_social)],
            ASK_SOCIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, finalize_log)],
        },
        fallbacks=[CommandHandler("start", start)]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(conv)
    app.add_handler(CommandHandler("score", score_cmd))
    app.add_handler(CommandHandler("week", week_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    return app

def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in .env")
    app = build_app(token)
    print("Axis bot is running. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
