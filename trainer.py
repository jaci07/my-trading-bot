import pandas as pd
import numpy as np
import os
import joblib
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Eigene Module
from infrastructure import AIEngine, log, DatabaseHandler, VolumeProfileEngine
from mt5_handler import MT5Handler
from advanced_engine import AdvancedMarketEngine
from settings import cfg

class StrategyAITrainer:
    def __init__(self):
        log.info("🛠️ Initialisiere Turbo-Trainer mit Strategie-Sync...")
        
        # 1. Erstelle die benötigten Abhängigkeiten
        self.mt5_handler = MT5Handler()
        self.db_handler = DatabaseHandler()
        self.vp_engine = VolumeProfileEngine()
        
        # 2. Initialisiere Engines (FIX: Übergabe der Handlers)
        self.ai_engine = AIEngine()
        self.strat_engine = AdvancedMarketEngine(self.mt5_handler, self.db_handler)
        
        # 3. Deine Feature-Liste (Kleingeschrieben für Kompatibilität)
        self.feature_list = [
            'rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2',
            'macd_hist', 'trend_strength', 'macd_hist_prev1', 'macd_hist_prev2',
            'bb_pct', 'bb_width', 'atr', 'mfi', 'obv_slope',
            'wick_upper', 'wick_lower', 'is_doji', 'engulfing'
        ]

    def simulate_outcome(self, df, start_idx, side):
        """Prüft, ob ein Signal in der Zukunft TP oder SL getroffen hätte."""
        lookahead = 48  # Prüfe die nächsten 4 Stunden (M5)
        subset = df.iloc[start_idx + 1 : start_idx + lookahead + 1]
        if subset.empty: return 0
        
        entry_price = df['close'].iloc[start_idx]
        atr = df['atr'].iloc[start_idx]
        
        tp = entry_price + (atr * 2.0) if side == "LONG" else entry_price - (atr * 2.0)
        sl = entry_price - (atr * 1.5) if side == "LONG" else entry_price + (atr * 1.5)
        
        for _, row in subset.iterrows():
            if side == "LONG":
                if row['high'] >= tp: return 1 # Win
                if row['low'] <= sl: return 2  # Loss
            else:
                if row['low'] <= tp: return 1  # Win
                if row['high'] >= sl: return 2 # Loss
        return 0 # Timeout/Neutral

    def train_all(self):
        for symbol in cfg.SYMBOLS:
            log.info(f"--- 🚀 Starte Training für {symbol} ---")
            
            # Daten laden (15.000 Kerzen für massive Datenbasis)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 15000)
            if rates is None: continue
            df_raw = pd.DataFrame(rates)
            
            # Features berechnen
            df = self.ai_engine.feature_engineering(df_raw)
            if df.empty: continue

            X_data = []
            y_data = []
            
            log.info(f"🔍 Scanne {len(df)} Kerzen nach Strategie-Signalen...")
            
            # Strategie-Scan (Wir lernen nur aus echten Signalen der AdvancedEngine!)
            for i in range(250, len(df) - 50):
                subset = df.iloc[i-200 : i+1]
                direction, _ = self.strat_engine.check_entry_signal(symbol, subset, self.vp_engine)
                
                if direction:
                    outcome = self.simulate_outcome(df, i, direction)
                    if outcome != 0:
                        # Wir speichern das Feature-Set und ob es ein Win (1) oder Loss (2) war
                        features = df.iloc[i][self.feature_list].values
                        X_data.append(features)
                        y_data.append(outcome)

            if len(X_data) < 20:
                log.warning(f"⚠️ Zu wenige Setups ({len(X_data)}) für {symbol}. Überspringe...")
                continue

            log.info(f"📊 {len(X_data)} Setups gefunden. Starte Deep-Learning Training...")

            # Das "Fleisch": Hohe Komplexität für große Dateien
            model = RandomForestClassifier(
                n_estimators=500, 
                max_depth=None, 
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_data, y_data)

            # Output-Statistiken
            train_acc = accuracy_score(y_data, model.predict(X_data))
            log.info(f"🎯 Training abgeschlossen. Genauigkeit: {train_acc:.2%}")

            # Speichern
            save_path = f"ai_models/{symbol}_model.pkl"
            if not os.path.exists("ai_models"): os.makedirs("ai_models")
            joblib.dump(model, save_path)
            
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            log.info(f"💾 Modell gespeichert: {size_mb:.2f} MB")

if __name__ == "__main__":
    trainer = StrategyAITrainer()
    trainer.train_all()
