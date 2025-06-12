import os, time, json, threading, random, winsound
import numpy as np, pandas as pd
import ta
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from tradingview_ta import TA_Handler, Interval, Exchange
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.linear_model import LogisticRegressionCV

CHROME_DRIVER_PATH = "chromedriver.exe"
POCKET_OPTION_URL = "https://pocketoption.com/en/cabinet/demo-quick-high-low/"
CANDLE_DURATION_SEC = 60.0
BEST_TRADE_OFFSET = 55.0
FEATURE_COUNT = 39
TRAINING_DATA_FILE = "training_data.json"
TRADE_LOG_FILE = "trade_log.json"
INITIAL_REWARD = 1.0

otc_symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # Symbols for TradingView
current_symbol_index = 0
model = LogisticRegressionCV(class_weight="balanced", max_iter=1000)
model_fitted = False
history = []
ROOT_TK = None
reward = INITIAL_REWARD
plot_data = []

class ProgressBarUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bot Progress")
        self.root.geometry("260x100+1200+100")
        self.root.attributes("-topmost", True)
        self.var = tk.DoubleVar()
        self.bar = tk.Scale(root, variable=self.var, from_=0, to=100,
                            orient=tk.HORIZONTAL, label="Model Accuracy %")
        self.bar.pack(pady=20)

    def update(self, acc: float):
        self.var.set(round(max(0.0, min(1.0, acc)) * 100))

class BirdAnimation:
    def __init__(self, root, image_path="bird.png"):
        try:
            img = Image.open(image_path)
            self.photo = ImageTk.PhotoImage(img)
            w, h = img.size
            win = tk.Toplevel(root)
            win.overrideredirect(True)
            win.config(bg="magenta")
            win.attributes("-transparentcolor", "magenta", "-topmost", True)
            win.geometry(f"{w}x{h}+{-w}+{int(win.winfo_screenheight() * 0.3)}")
            canvas = tk.Canvas(win, width=w, height=h, bg="magenta", highlightthickness=0)
            canvas.pack()
            canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.win, self.w, self.h = win, w, h
            self._animate()
        except Exception as e:
            print(f"🐦 Animation failed: {e}")

    def _animate(self, elapsed=0, duration=2000):
        if elapsed >= duration:
            self.win.destroy(); return
        x = int(-self.w + (self.win.winfo_screenwidth() + self.w) * (elapsed / duration))
        self.win.geometry(f"{self.w}x{self.h}+{x}+{self.win.winfo_y()}")
        self.win.after(20, self._animate, elapsed + 20, duration)

def setup_driver():
    opts = Options()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(CHROME_DRIVER_PATH), options=opts)

def get_tradingview_closes(symbol):
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener="forex",
            exchange="FX_IDC",
            interval=Interval.INTERVAL_1_MINUTE,
        )
        candles = handler.get_analysis().indicators
        closes = candles.get("close", [])
        if isinstance(closes, list):
            return closes[-25:]
        return []
    except Exception as e:
        print(f"⚠️ TradingView error: {e}")
        return []

def extract_features(symbol):
    closes = get_tradingview_closes(symbol)
    if len(closes) < 25:
        print(f"⚠️ Not enough candles for {symbol}")
        return None
    df = pd.DataFrame({"close": closes})
    df["ema"] = ta.trend.ema_indicator(df["close"], window=5)
    df["rsi"] = ta.momentum.rsi(df["close"], window=5)
    if df[["ema", "rsi"]].isna().any().any():
        return None
    return [
        df["ema"].iat[-1] - df["ema"].iat[-2],
        df["rsi"].iat[-1],
        df["close"].iat[-1],
        df["ema"].iat[-1],
    ] + list(np.random.rand(FEATURE_COUNT - 4))

def predict(features):
    if not model_fitted:
        return random.choice(["BUY", "SELL"]), 0.5
    try:
        probs = model.predict_proba([features])[0]
        idx = int(np.argmax(probs))
        return ("SELL" if model.classes_[idx] == 1 else "BUY", float(probs[idx]))
    except:
        return random.choice(["BUY", "SELL"]), 0.5

def execute_trade(driver, decision):
    xpath = "//a[contains(@class,'btn-call')]" if decision == "BUY" else "//a[contains(@class,'btn-put')]"
    try:
        btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        btn.click()
        print(f"✅ Executed {decision}")
    except Exception as e:
        print(f"❌ Trade failed: {e}")

def load_training_data():
    global model_fitted
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, "r") as f:
            for h in json.load(f):
                if "features" in h and isinstance(h["features"], list) and len(h["features"]) == FEATURE_COUNT:
                    history.append(h)

    # Ensure at least one of each class
    class_counts = {0: 0, 1: 0}
    for h in history:
        class_counts[h["outcome"]] += 1

    while class_counts[0] < 1 or class_counts[1] < 1:
        outcome = 0 if class_counts[0] <= class_counts[1] else 1
        history.append({
            "features": list(np.random.rand(FEATURE_COUNT)),
            "outcome": outcome
        })
        class_counts[outcome] += 1

    # 🚨 Final filter before model.fit
    filtered = [h for h in history if isinstance(h["features"], list) and len(h["features"]) == FEATURE_COUNT]
    X = [h["features"] for h in filtered]
    y = [h["outcome"] for h in filtered]

    model.fit(X, y)
    model_fitted = True

def save_training_data():
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(history, f, indent=2)

def log_trade(entry):
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "r") as f:
            trades = json.load(f)
    trades.append(entry)
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(trades, f, indent=2)

def bot_loop(pb):
    global reward, current_symbol_index
    load_training_data()
    driver = setup_driver()
    driver.get(POCKET_OPTION_URL)
    input("🔐 Log in and press ENTER.")
    input("📈 Select an asset and press ENTER.")

    while True:
        symbol = otc_symbols[current_symbol_index]
        print(f"\n🚀 Analyzing: {symbol}")

        now = datetime.now(timezone.utc)
        delay = CANDLE_DURATION_SEC - (now.second + now.microsecond / 1e6) % CANDLE_DURATION_SEC
        time.sleep(delay)
        time.sleep(BEST_TRADE_OFFSET)

        feats = extract_features(symbol)
        if feats is None:
            current_symbol_index = (current_symbol_index + 1) % len(otc_symbols)
            continue

        dec, conf = predict(feats)
        print(f"📊 {dec} | Confidence: {conf:.2f}")

        if conf > 0.70:
            winsound.Beep(1000, 300)
            execute_trade(driver, dec)

        closes = get_tradingview_closes(symbol)
        if not closes: continue
        exit_price = closes[-1]
        entry_price = feats[2]
        outcome = int((dec == "BUY" and exit_price > entry_price) or (dec == "SELL" and exit_price < entry_price))

        history.append({"features": feats, "outcome": outcome})
        log_trade({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "decision": dec,
            "confidence": conf,
            "entry": entry_price,
            "exit": exit_price,
            "outcome": outcome
        })

        if outcome == 1 and ROOT_TK:
            try: ROOT_TK.after(0, lambda: BirdAnimation(ROOT_TK))
            except: pass

        X = [h["features"] for h in history]
        y = [h["outcome"] for h in history]
        reward = min(reward * 1.1, 10.0) if outcome == 1 else max(reward * 0.95, 0.1)

        if len(history) >= 10:
            sw = np.array([reward if o == 1 else 1.0 for o in y])
            model.fit(X, y, sample_weight=sw)
            save_training_data()
            try:
                acc = model.score(X, y)
                pb.update(acc)
                plot_data.append(acc)
                if len(plot_data) % 5 == 0:
                    plt.clf()
                    plt.plot(plot_data, label="Accuracy")
                    plt.ylim(0, 1)
                    plt.title("Model Accuracy Over Time")
                    plt.legend()
                    plt.pause(0.1)
            except:
                pass

        current_symbol_index = (current_symbol_index + 1) % len(otc_symbols)

if __name__ == "__main__":
    ROOT_TK = tk.Tk()
    bar = ProgressBarUI(ROOT_TK)
    plt.ion()
    plt.show()
    threading.Thread(target=bot_loop, args=(bar,), daemon=True).start()
    ROOT_TK.mainloop()
