
# Axis Life Score — Telegram MVP

Minimal, working MVP to compute **Life Score** via a Telegram bot.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# put your TELEGRAM_BOT_TOKEN into .env
python axis_bot.py
```

Create a bot token via Telegram's @BotFather:
- `/newbot` → choose name + username
- copy the token → put it into `.env`

## 2) Commands

- `/start` — intro
- `/log` — enter today's values (mood, sleep hours, activity 0–10, focus 0–10, social 0–10)
- `/score` — compute and show today's Life Score
- `/week` — average Life Score over last 7 days
- `/export` — dump your raw JSON data

## 3) Model (MVP)

- **Mood (core)** = α·M_eff + (1−α)·M_hat, where
  - M_eff — lagged normalized mood from last 3 days
  - M_hat — proxy from Stability (same scale)
  - α = 0.7

- **Stability (basic layer)** = Σ (w_i × X_i_eff_final):
  - factors: sleep, activity, focus, social
  - w = {sleep 0.35, focus 0.30, activity 0.20, social 0.15}
  - per-factor normalization uses **personal sweet spots**, derived from history:
    - optimal = mean of days with mood >= median over last 14 days (fallback defaults)
    - tolerance = max(1.5, std_dev)
  - lag: 0.6 today, 0.3 yesterday, 0.1 two days ago

- **Life Score** = 100 × [ β·Mood + (1−β)·Stability ]
  - β = 0.65
  - banding: <50 🔴, 50–74 🟡, 75+ 🟢

Everything is stored locally under `data/user_<id>.json`.

## 4) Notes
- Missing values for a day default to 0 for normalization (can be refined).
- This code is intentionally simple and readable to allow fast iteration.
- You can tweak α/β/weights in `axis_bot.py`.
