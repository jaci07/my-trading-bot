import warnings
warnings.filterwarnings("ignore")
# infrastructure.py
import logging
import joblib
import sqlite3
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import time
import pickle
import sys
from datetime import datetime, timedelta
from colorama import init, Fore, Style
from sklearn.ensemble import RandomForestClassifier
from settings import cfg
# Unterdrückt die nervigen Parallel-Warnungen
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Optional: Unterdrückt TensorFlow/System Warnungen falls vorhanden
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. LOGGING SYSTEM ---
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': Fore.CYAN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'DEBUG': Fore.GREEN
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

log = logging.getLogger("EnterpriseBot")
log.setLevel(logging.DEBUG)
if not log.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(ch)
    fh = logging.FileHandler("bot_activity.log", encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    log.addHandler(fh)

# --- 2. DATENBANK HANDLER ---
class DatabaseHandler:
    def __init__(self):
        self.db_path = cfg.DB_NAME
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()
        self.update_schema()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, side TEXT, qty REAL, price REAL,
                setup TEXT, features TEXT, result REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN', ticket_id INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def update_schema(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(trades)")
            cols = [info[1] for info in cursor.fetchall()]
            if 'ticket_id' not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN ticket_id INTEGER DEFAULT 0")
            self.conn.commit()
        except: pass

    def log_trade(self, symbol, side, qty, price, setup, features_dict=None, ticket_id=0):
        import json
        f_json = json.dumps(features_dict) if features_dict else "{}"
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO trades (symbol, side, qty, price, setup, features, status, ticket_id) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)",
                       (symbol, side, float(qty), float(price), setup, f_json, int(ticket_id)))
        self.conn.commit()
        return cursor.lastrowid

    def has_traded_today(self, symbol, setup_type):
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM trades WHERE symbol=? AND setup LIKE ? AND date(timestamp) = date('now')", (symbol, f"%{setup_type}%"))
        return cursor.fetchone()[0] > 0

    def get_minutes_since_last_trade(self, symbol):
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp FROM trades WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (symbol,))
        row = cursor.fetchone()
        if row:
            try:
                diff = datetime.now() - datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                return diff.total_seconds() / 60
            except: return 9999
        return 9999

# --- 3. VOLUME PROFILE ENGINE ---
class VolumeProfileEngine:
    def __init__(self):
        self.poc = None
        self.vah = None
        self.val = None
        self.profile_data = None

    def calculate_enhanced_profile(self, df, lookback=96, decay=0.95):
        if df is None or len(df) < 20: return 0,0,0
        subset = df.iloc[-lookback:].copy()
        
        # Robustes Volumen-Handling
        vol_col = next((c for c in ['tick_volume', 'volume', 'real_volume'] if c in subset.columns), None)
        if not vol_col:
            subset['dummy_vol'] = 1
            vol_col = 'dummy_vol'

        weights = [decay ** i for i in range(len(subset))]
        weights.reverse()
        subset['weighted_vol'] = subset[vol_col] * weights
        
        hist, bin_edges = np.histogram(subset['close'], bins=50, weights=subset['weighted_vol'])
        hist_smooth = pd.Series(hist).rolling(window=3, center=True, min_periods=1).mean().fillna(0).values
        self.profile_data = pd.DataFrame({'vol': hist_smooth, 'price': bin_edges[:-1]})
        
        self.poc = self.profile_data.loc[self.profile_data['vol'].idxmax(), 'price']
        total_v = self.profile_data['vol'].sum()
        va_v = total_v * 0.70
        sorted_p = self.profile_data.sort_values(by='vol', ascending=False)
        sorted_p['cum_v'] = sorted_p['vol'].cumsum()
        va_bins = sorted_p[sorted_p['cum_v'] <= va_v]
        
        if not va_bins.empty:
            self.vah, self.val = va_bins['price'].max(), va_bins['price'].min()
        else:
            self.vah = self.val = self.poc
        return self.poc, self.vah, self.val

    def find_nearest_lva(self, df, current_price, direction="DOWN"):
        if self.profile_data is None: return None
        threshold = self.profile_data['vol'].mean() * 0.40
        if direction == "DOWN":
            cands = self.profile_data[self.profile_data['price'] < current_price].sort_values(by='price', ascending=False)
        else:
            cands = self.profile_data[self.profile_data['price'] > current_price].sort_values(by='price', ascending=True)
        for _, r in cands.iterrows():
            if r['vol'] < threshold: return r['price']
        return None

    def find_last_pivot(self, df):
        """
        Smarter Anchor: Looks back a full 24h (300 candles on M5) to find 
        the true major structural swing high or low for the Volume Profile.
        """
        if df is None or len(df) < 50:
            return df.index[0] if df is not None and not df.empty else 0
            
        # 1. Expand lookback to ~1 full trading day (300 candles on M5)
        lookback = min(300, len(df))
        recent_df = df.iloc[-lookback:]
        
        # 2. Find the absolute structural extremes in this broader window
        highest_idx = recent_df['high'].idxmax()
        lowest_idx = recent_df['low'].idxmin()
        
        # 3. Anchor at the start of the CURRENT major move
        anchor_idx = min(highest_idx, lowest_idx)
        
        # 4. THE FIX: Count the actual rows from the anchor to the end.
        # This completely avoids the "int vs Timestamp" math error!
        if len(df.loc[anchor_idx:]) < 20:
            anchor_idx = max(highest_idx, lowest_idx)
            
        return anchor_idx

    def calculate_vwap(self, df):
        """
        Berechnet den Volume Weighted Average Price (VWAP) für das übergebene DataFrame.
        """
        if df is None or len(df) == 0:
            return None
            
        # Typischer Preis: (High + Low + Close) / 3
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Welches Volumen haben wir? (tick_volume ist im MT5 Standard)
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        
        # Falls absolut kein Volumen im Chart ist (sehr selten), Fallback auf normalen Preis
        if vol_col not in df.columns:
            return typical_price.iloc[-1]
            
        volume = df[vol_col]
        
        # VWAP Formel: Kumuliertes (Preis * Volumen) / Kumuliertes Volumen
        cumulative_vp = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        # Um eine Division durch 0 zu vermeiden
        cumulative_volume = cumulative_volume.replace(0, 1)
        
        vwap = cumulative_vp / cumulative_volume
        
        # Wir geben den aktuellsten (letzten) VWAP-Wert zurück
        return vwap.iloc[-1]
# --- 4. AI ENGINE ---
class AIEngine:
    def __init__(self):
        self.models_dir = "ai_models"
        self.models = {}
        if not os.path.exists(self.models_dir): os.makedirs(self.models_dir)

    def feature_engineering(self, df):
        df = df.copy()
        try:
            df.columns = [c.lower() for c in df.columns]
            vol_col = next((c for c in ['tick_volume', 'volume', 'real_volume'] if c in df.columns), None)
            
            # Indikatoren
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            stoch = ta.stochrsi(df['close'])
            df['stoch_k'] = stoch.iloc[:, 0] if stoch is not None else 50
            macd = ta.macd(df['close'])
            df['macd_hist'] = macd.iloc[:, 1] if macd is not None else 0
            df['ema_20'], df['ema_50'] = ta.ema(df['close'], 20), ta.ema(df['close'], 50)
            df['trend_strength'] = df['ema_20'] - df['ema_50']
            bb = ta.bbands(df['close'])
            if bb is not None:
                df['bb_pct'] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
                df['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']
            
            if vol_col:
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df[vol_col], length=14)
                df['obv_slope'] = ta.obv(df['close'], df[vol_col]).diff(5)

            for c in ['rsi', 'macd_hist', 'trend_strength']:
                df[f'{c}_prev1'], df[f'{c}_prev2'] = df[c].shift(1), df[c].shift(2)
            
            df['wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_doji'] = np.where(abs(df['close']-df['open']) <= (df['high']-df['low'])*0.1, 1, 0)
            df['engulfing'] = 0 # Platzhalter
            
            df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
            return df
        except Exception as e:
            log.error(f"Feature Error: {e}")
            return pd.DataFrame()

    def get_prediction_proba_all(self, symbol, df, tf_name="M5"):
        model_key = f"{symbol}_{tf_name}"
        model = self.models.get(model_key)
        if model is None:
            fn = os.path.join(self.models_dir, f"{model_key}_model.pkl")
            if os.path.exists(fn):
                try: 
                    model = joblib.load(fn)
                    model.n_jobs = 1
                    self.models[symbol] = model
                except: return [1.0, 0.0, 0.0]
            else: return [1.0, 0.0, 0.0]

        try:
            data = self.feature_engineering(df)
            feats = ['rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2', 'macd_hist', 'trend_strength', 
                     'macd_hist_prev1', 'macd_hist_prev2', 'bb_pct', 'bb_width', 'atr', 'mfi', 
                     'obv_slope', 'wick_upper', 'wick_lower', 'is_doji', 'engulfing']
            
            X = data[feats].iloc[-1].values.reshape(1, -1)
            probs = model.predict_proba(X)[0]
            
            # WICHTIG: Mapping für 3 Klassen (0=Nix, 1=Win/Long, 2=Loss/Short)
            if len(probs) == 3:
                return [probs[0], probs[1], probs[2]]
            elif len(probs) == 2:
                return [0.0, probs[0], probs[1]]
            return [1.0, 0.0, 0.0]
        except: return [1.0, 0.0, 0.0]

    def get_ai_prediction(self, symbol, df, tf_name="M5"):
        # tf_name wird weitergereicht, damit das richtige Modell (M1 oder M5) geladen wird
        p = self.get_prediction_proba_all(symbol, df, tf_name)
        return {"nix": p[0], "long": p[1], "short": p[2]}

    def get_prediction_prob(self, symbol, df):
        return self.get_ai_prediction(symbol, df)["long"]

    def save_experience(self, symbol, features, label):
        fp = os.path.join(self.models_dir, "smart_memory.csv")
        data = features.copy(); data['symbol'] = symbol; data['Target'] = label
        df_new = pd.DataFrame([data])
        df_new.to_csv(fp, mode='a', header=not os.path.exists(fp), index=False)
