# sweet_spots.py
import pandas as pd
from datetime import datetime
import json
import os
import logging
from pathlib import Path

# === Налаштування збереження файлів (працює локально і на Render) ===
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = str(DATA_DIR / "logs.csv")
SWEET_FILE = str(DATA_DIR / "sweet_spots.json")

# === Логування в консоль (видно в Render Logs) ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === ЗБЕРЕЖЕННЯ ЩОДЕННОГО ЛОГУ КОРИСТУВАЧА ===
def save_log(user_id, mood, sleep, activity, focus, social):
    new_data = pd.DataFrame([{
        "user_id": str(user_id),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mood": float(mood),
        "sleep": float(sleep),
        "activity": float(activity),
        "focus": float(focus),
        "social": float(social)
    }])

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(LOG_FILE, index=False)
    logging.info(f"[{user_id}] Logged entry: mood={mood}, sleep={sleep}, activity={activity}, focus={focus}, social={social}")

# === ДОСТУП ДО ІСТОРІЇ ТА ПОТОЧНИХ SWEET SPOTS ===
def get_user_logs(user_id):
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    return df[df['user_id'] == str(user_id)]

def _load_sweet_spots():
    if os.path.exists(SWEET_FILE):
        with open(SWEET_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_sweet_spots(data):
    with open(SWEET_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_sweet_spots(user_id):
    data = _load_sweet_spots()
    return data.get(str(user_id))

# === АНАЛІТИКА: ПОШУК ОПТИМУМУ ТА ОНОВЛЕННЯ SWEET SPOTS ===
def find_optimal_sweet_spot(df, factor):
    # середній настрій для кожного значення фактора
    grouped = df.groupby(factor)['mood'].mean()
    # якщо даних замало або NaN — повертаємо поточне середнє
    if grouped.empty:
        return float(df[factor].mean())
    return float(grouped.idxmax())

def update_sweet_spot(old, optimal, alpha=0.3):
    return float(old) * (1 - alpha) + float(optimal) * alpha

def apply_tolerance(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def recalculate_all_sweet_spots(user_id):
    if not os.path.exists(LOG_FILE):
        logging.info(f"[{user_id}] No logs yet; skip sweet spot update.")
        return

    df = pd.read_csv(LOG_FILE)
    df = df[df['user_id'] == str(user_id)]

    if len(df) < 10:
        logging.info(f"[{user_id}] Not enough data to update sweet spots ({len(df)}/10).")
        return

    # здорові межі (запобігають сповзанню в шкідливу «норму»)
    tolerances = {
        'sleep': (6.0, 9.0),
        'activity': (2.0, 10.0),
        'focus': (2.0, 10.0),
        'social': (0.0, 10.0)
    }

    # поточні sweet spots або дефолт
    sweet_spots = _load_sweet_spots()
    uid = str(user_id)
    if uid not in sweet_spots:
        sweet_spots[uid] = {
            'sleep': 8.0,
            'activity': 5.0,
            'focus': 6.0,
            'social': 3.0
        }

    new_spots = {}
    for factor in ['sleep', 'activity', 'focus', 'social']:
        optimal = find_optimal_sweet_spot(df, factor)
        prev = float(sweet_spots[uid][factor])
        updated = update_sweet_spot(prev, optimal, alpha=0.3)
        bounded = apply_tolerance(updated, *tolerances[factor])
        new_spots[factor] = round(bounded, 2)
        logging.info(f"[{user_id}] {factor.capitalize()} sweet spot: {prev:.2f} → {new_spots[factor]:.2f} (optimal {optimal:.2f})")

    sweet_spots[uid] = new_spots
    _save_sweet_spots(sweet_spots)
    logging.info(f"[{user_id}] ✅ Sweet spots updated: {new_spots}")
