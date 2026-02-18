# infrastructure.py
import logging
import joblib
import sqlite3
import pandas as pd
import pandas_ta as ta # Falls du pandas_ta nutzt, sonst kann das weg
import numpy as np
import os
import time
import pickle
from datetime import datetime, timedelta
from colorama import init, Fore, Style
from sklearn.ensemble import RandomForestClassifier
from settings import cfg
import sys

# --- 1. LOGGING SYSTEM (Windows Safe) ---
# Windows Konsole auf UTF-8 zwingen (WICHTIG!)
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

# Logger einrichten
log = logging.getLogger("EnterpriseBot")
log.setLevel(logging.DEBUG)

# Konsole Handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# Datei Handler
fh = logging.FileHandler("bot_activity.log", encoding='utf-8')
fh.setLevel(logging.INFO)
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
log.addHandler(fh)

# --- 2. DATENBANK HANDLER (Mit Auto-Update) ---
class DatabaseHandler:
    def __init__(self):
        self.db_path = cfg.DB_NAME
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()
        self.update_schema() # <--- DAS REPARIERT DEINE DB

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                setup TEXT,
                features TEXT, 
                result REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_models (
                symbol TEXT PRIMARY KEY,
                last_trained DATETIME,
                accuracy REAL
            )
        ''')
        self.conn.commit()

    def update_schema(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(trades)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'features' not in columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN features TEXT")
            if 'result' not in columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN result REAL DEFAULT 0")
            if 'status' not in columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN status TEXT DEFAULT 'OPEN'")
            
            # --- NEU: TICKET ID ---
            if 'ticket_id' not in columns:
                log.warning("🛠️ Datenbank-Update: Füge Spalte 'ticket_id' hinzu...")
                cursor.execute("ALTER TABLE trades ADD COLUMN ticket_id INTEGER DEFAULT 0")
                
            self.conn.commit()
        except Exception as e:
            log.error(f"Fehler beim DB-Update: {e}")

    def log_trade(self, symbol, side, qty, price, setup, features_dict=None, ticket_id=0):
        import json
        if features_dict is None: features_dict = {}
        features_json = json.dumps(features_dict)
        
        cursor = self.conn.cursor()
        # Wir speichern jetzt auch die TICKET ID!
        cursor.execute("""
            INSERT INTO trades (symbol, side, qty, price, setup, features, status, ticket_id) 
            VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (symbol, side, float(qty), float(price), setup, features_json, int(ticket_id)))
        
        self.conn.commit()
        log.info(f"💾 Trade {ticket_id} in DB gespeichert: {symbol}")
        return cursor.lastrowid

    def has_traded_today(self, symbol, setup_type):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT count(*) FROM trades 
            WHERE symbol=? AND setup LIKE ? AND date(timestamp) = date('now')
        ''', (symbol, f"%{setup_type}%"))
        count = cursor.fetchone()[0]
        return count > 0

    def get_minutes_since_last_trade(self, symbol):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp FROM trades 
            WHERE symbol=? ORDER BY timestamp DESC LIMIT 1
        ''', (symbol,))
        row = cursor.fetchone()
        if row:
            try:
                last_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                diff = datetime.now() - last_time
                return diff.total_seconds() / 60
            except: return 9999
        return 9999


# infrastructure.py - VolumeProfileEngine UPDATE

class VolumeProfileEngine:
    def __init__(self):
        self.poc = None
        self.vah = None
        self.val = None
        self.profile_data = None 

    def calculate_vwap(self, df):
        if df is None or df.empty: return 0.0
        # Fallback falls kein 'volume', nutze 'tick_volume'
        vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df[vol_col]
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap.iloc[-1]


    def find_last_pivot(self, df, lookback=30):
        """
        Findet den Anker-Punkt (Start des Trends) für das Volume Profile.
        Sucht das tiefste Low (oder höchste High) der letzten 'lookback' Kerzen.
        """
        if df is None or len(df) < lookback: return df.index[0]
        
        # Wir suchen vereinfacht das tiefste Low der letzten Bewegung als Startpunkt
        # (Ideal für Long-Einstiege nach Pullback)
        subset = df['low'].iloc[-lookback:]
        lowest_idx = subset.idxmin()
        
        return lowest_idx

    def calculate_enhanced_profile(self, df, lookback=96, decay=0.95):
        """
        LOGIK-UPDATE (Artikel):
        1. Nur kurze Historie (z.B. 96 Kerzen = 24h)
        2. 'Decay': Neue Kerzen zählen mehr als alte.
        3. 'Smoothing': Glättet das Profil.
        """
        if df is None or len(df) < 20: return 0,0,0

        subset = df.iloc[-lookback:].copy()
        
        # 1. Gewichtung berechnen (Exponentiell)
        # Die neueste Kerze hat Gewicht 1.0, die davor 0.95, davor 0.90...
        weights = [decay ** i for i in range(len(subset))]
        weights.reverse() # Umdrehen: Älteste zuerst, Neueste zuletzt (1.0)
        
        subset['weighted_vol'] = subset['volume'] * weights

        # 2. Histogramm erstellen
        bins = 50 # Weniger Bins = Mehr "Cluster" (Besser für Zonen)
        hist, bin_edges = np.histogram(subset['close'], bins=bins, weights=subset['weighted_vol'])
        
        # 3. Glättung (Smoothing) - Einfacher gleitender Durchschnitt über das Histogramm
        # Das entfernt kleine "Zacken" und findet echte Berge
        hist_smooth = pd.Series(hist).rolling(window=3, center=True, min_periods=1).mean().fillna(0).values

        self.profile_data = pd.DataFrame({'vol': hist_smooth, 'price': bin_edges[:-1]})
        
        # 4. POC finden (im geglätteten Profil)
        max_vol_idx = self.profile_data['vol'].idxmax()
        self.poc = self.profile_data.loc[max_vol_idx, 'price']
        
        # 5. Value Area (70%)
        total_volume = self.profile_data['vol'].sum()
        value_area_vol = total_volume * 0.70
        
        sorted_prof = self.profile_data.sort_values(by='vol', ascending=False)
        sorted_prof['cum_vol'] = sorted_prof['vol'].cumsum()
        
        va_bins = sorted_prof[sorted_prof['cum_vol'] <= value_area_vol]
        
        if not va_bins.empty:
            self.vah = va_bins['price'].max()
            self.val = va_bins['price'].min()
        else:
            self.vah = self.poc
            self.val = self.poc

        return self.poc, self.vah, self.val

    # Wrapper damit der alte Code in main.py nicht kaputt geht
    def calculate_frvp(self, df):
        return self.calculate_enhanced_profile(df)

    def find_nearest_lva(self, df, current_price, direction="DOWN"):
        # Nutzt jetzt das geglättete Profil -> Findet bessere "Lücken"
        if self.profile_data is None or self.profile_data.empty:
            self.calculate_enhanced_profile(df)
            
        profile = self.profile_data
        if profile is None or profile.empty: return None

        avg_vol = profile['vol'].mean()
        # Alles unter 40% des Durchschnitts ist eine Lücke (LVA)
        threshold = avg_vol * 0.40 
        
        if direction == "DOWN":
            candidates = profile[profile['price'] < current_price]
            candidates = candidates.sort_values(by='price', ascending=False) # Nächste unter uns
            for _, row in candidates.iterrows():
                if row['vol'] < threshold: return row['price']
                    
        elif direction == "UP":
            candidates = profile[profile['price'] > current_price]
            candidates = candidates.sort_values(by='price', ascending=True) # Nächste über uns
            for _, row in candidates.iterrows():
                if row['vol'] < threshold: return row['price']
        
        return None


# --- 4. AI ENGINE (Das Gehirn) ---
class AIEngine:
    def __init__(self):
        self.models_dir = "ai_models"
        self.models = {}
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def feature_engineering(self, df):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        try:
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
            df['bb_pct'] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0]) if bb is not None else 0.5
            df['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close'] if bb is not None else 0
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['tick_volume'])
            df['obv_slope'] = ta.obv(df['close'], df['tick_volume']).diff(5)
            for c in ['rsi', 'macd_hist', 'trend_strength']:
                df[f'{c}_prev1'], df[f'{c}_prev2'] = df[c].shift(1), df[c].shift(2)
            df['wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_doji'] = np.where(abs(df['close']-df['open']) <= (df['high']-df['low'])*0.1, 1, 0)
            df['engulfing'] = 0 # Vereinfacht
            df.ffill(inplace=True); df.bfill(inplace=True); df.fillna(0, inplace=True)
            return df
        except Exception as e:
            log.error(f"Feature Error: {e}"); return pd.DataFrame()

    # ---------------------------------------------------------
    # 2. Train Models (Das Training anpassen)
    # ---------------------------------------------------------
    def train_models(self, symbol, df):
        try:
            df = self.feature_engineering(df)
            if len(df) < 100: return 
            
            # WICHTIG: Die Features müssen exakt so heißen wie die Spalten (kleingeschrieben!)
            features = [
                'rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2',
                'macd_hist', 'trend_strength', 'macd_hist_prev1', 'macd_hist_prev2',
                'trend_strength_prev1', 'trend_strength_prev2',
                'bb_pct', 'bb_width', 'atr',
                'mfi', 'obv_slope', 'mfi_prev1', 'mfi_prev2',
                'wick_upper', 'wick_lower', 'is_doji', 'engulfing'
            ]
            
            # Prüfen welche Features wirklich im DF sind
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                log.warning(f"⚠️ Keine Features für {symbol} gefunden. Training abgebrochen.")
                return

            # Target: Wir brauchen 3 Klassen für deine get_ai_prediction Logik!
            # 0 = Seitwärts/Nix, 1 = Long Win, 2 = Short Win
            # Hier eine einfache Logik (sollte später durch Simulation ersetzt werden):
            df['target'] = 0
            # Wenn Preis um mehr als 0.1% steigt -> Long
            df.loc[df['close'].shift(-10) > df['close'] * 1.001, 'target'] = 1
            # Wenn Preis um mehr als 0.1% fällt -> Short
            df.loc[df['close'].shift(-10) < df['close'] * 0.999, 'target'] = 2

            X = df[available_features]
            y = df['target']
            
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X, y)

            self.models[symbol] = model
            
            # Speichern mit joblib (konsistent zum Laden)
            filename = os.path.join(self.models_dir, f"{symbol}_model.pkl")
            joblib.dump(model, filename)
            
            log.info(f"🧠 Modell für {symbol} erfolgreich trainiert und gespeichert.")
            
        except Exception as e:
            log.error(f"❌ Training Error {symbol}: {e}")

    def get_ai_prediction(self, symbol, df):
        probs = self.get_prediction_proba_all(symbol, df)
        return {"nix": probs[0], "long": probs[1], "short": probs[2]}

    def get_prediction_proba_all(self, symbol, df):
        """Lädt das Modell und berechnet Wahrscheinlichkeiten [Neutral, Win, Loss]"""
        model = self.models.get(symbol)
        if model is None:
            filename = os.path.join(self.models_dir, f"{symbol}_model.pkl")
            if os.path.exists(filename):
                try:
                    model = joblib.load(filename)
                    self.models[symbol] = model
                except: return [1.0, 0.0, 0.0]
            else: return [1.0, 0.0, 0.0]

        try:
            data = self.feature_engineering(df)
            features = ['rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2', 'macd_hist', 'trend_strength', 
                        'macd_hist_prev1', 'macd_hist_prev2', 'bb_pct', 'bb_width', 'atr', 'mfi', 
                        'obv_slope', 'wick_upper', 'wick_lower', 'is_doji', 'engulfing']
            
            # Warnung verhindern durch .values.reshape
            X = data[features].iloc[-1].values.reshape(1, -1)
            probs = model.predict_proba(X)[0]
            
            # Mapping für Trainer (1=Win, 2=Loss)
            # Falls Modell nur 1 Klasse kennt (sehr selten)
            if len(probs) == 1: return [0.0, 1.0, 0.0] 
            # Normalfall: probs[0] ist Win (1), probs[1] ist Loss (2)
            return [0.0, probs[0], probs[1]]
        except:
            return [1.0, 0.0, 0.0]

    def get_prediction_prob(self, symbol, df):
        return self.get_ai_prediction(symbol, df)["long"]
        
    # Füge das zur AIEngine Klasse hinzu
    def save_experience(self, symbol, features, label):
        """
        Speichert einen echten Trade als Trainingsdaten.
        Label: 1 = Win, 0 = Loss
        """
        file_path = os.path.join(self.models_dir, "smart_memory.csv")
        
        # Features ist ein Dictionary. Wir machen daraus eine Zeile.
        data = features.copy()
        data['symbol'] = symbol
        data['Target'] = label # Das ist, was die AI lernen soll
        
        df_new = pd.DataFrame([data])
        
        # Anfügen an die Datei (oder neu erstellen)
        if os.path.exists(file_path):
            df_new.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(file_path, mode='w', header=True, index=False)
            
        log.info(f"🧠 Erfahrung gespeichert: {symbol} -> {'WIN' if label==1 else 'LOSS'}")
