import pandas as pd
import numpy as np
np.NaN = np.nan
import os
import joblib
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # NEU: Für echten Test

# Eigene Module
from infrastructure import AIEngine, log, DatabaseHandler, VolumeProfileEngine
from mt5_handler import MT5Handler
from advanced_engine import AdvancedMarketEngine
from settings import cfg

class StrategyAITrainer:
    def __init__(self):
        log.info("🛠️ Initialisiere Turbo-Trainer mit Validierungs-Logik...")
        self.mt5_handler = MT5Handler()
        self.db_handler = DatabaseHandler()
        self.vp_engine = VolumeProfileEngine()
        self.ai_engine = AIEngine()
        self.strat_engine = AdvancedMarketEngine(self.mt5_handler, self.db_handler)
        
        self.feature_list = [
            'rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2',
            'macd_hist', 'trend_strength', 'macd_hist_prev1', 'macd_hist_prev2',
            'bb_pct', 'bb_width', 'atr', 'mfi', 'obv_slope',
            'wick_upper', 'wick_lower', 'is_doji', 'engulfing'
        ]

    def simulate_outcome(self, df, start_idx, side):
        lookahead = 48 
        subset = df.iloc[start_idx + 1 : start_idx + lookahead + 1]
        if subset.empty: return 0
        entry = df['close'].iloc[start_idx]
        atr = df['atr'].iloc[start_idx]
        tp = entry + (atr * 2.5) if side == "LONG" else entry - (atr * 2.5)
        sl = entry - (atr * 1.5) if side == "LONG" else entry + (atr * 1.5)
        for _, row in subset.iterrows():
            if side == "LONG":
                if row['high'] >= tp: return 1 
                if row['low'] <= sl: return 2  
            else:
                if row['low'] <= tp: return 1  
                if row['high'] >= sl: return 2 
        return 0 

    def train_all(self):
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15
        }
        for symbol in cfg.SYMBOLS:
            mt5.symbol_select(symbol, True)
            
            for tf_name, tf_value in timeframes.items():
                log.info(f"--- 🚀 Deep-Training: {symbol} auf {tf_name} ---")
                
                # Daten laden (Für M1 laden wir mehr, damit er genug Setups findet)
                anzahl_kerzen = 50000 if tf_name == "M1" else 50000
                rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, anzahl_kerzen)
                
                if rates is None or len(rates) == 0:
                    log.error(f"❌ Keine {tf_name} Daten für {symbol} erhalten.")
                    continue
                
                # =================================================================
                # FIX: Alles ab hier ist jetzt EINGERÜCKT, damit es für JEDEN 
                # Timeframe ausgeführt wird (und nicht nur für den letzten!)
                # =================================================================
                
                df = self.ai_engine.feature_engineering(pd.DataFrame(rates))
                if df.empty:
                    log.error(f"❌ Feature Engineering fehlgeschlagen für {symbol} auf {tf_name}")
                    continue

                X_data, y_data = [], []
                
                # FIX: M1 braucht einen größeren Rückblick für das Volumen-Profil, sonst stürzt Pandas ab!
                lookback = 400 if tf_name == "M1" else 200
                
                for i in range(lookback + 50, len(df) - 50):
                    subset = df.iloc[i-lookback : i+1]
                    direction, _ = self.strat_engine.check_entry_signal(symbol, subset, self.vp_engine)
                    
                    if direction:
                        outcome = self.simulate_outcome(df, i, direction)
                        if outcome != 0:
                            X_data.append(df.iloc[i][self.feature_list].values)
                            y_data.append(outcome)

                # --- DEBUG INFO ---
                if len(X_data) < 50:
                    log.warning(f"⚠️ Zu wenig Setups auf {tf_name} gefunden ({len(X_data)}/50 benötigt).")
                    continue
                # ------------------

                # --- NEU: DATEN SPLITTEN (80% Lernen, 20% Blind-Test) ---
                X_train, X_test, y_train, y_test = train_test_split(
                    X_data, y_data, test_size=0.2, random_state=42, shuffle=False
                )

                log.info(f"🧠 Training mit {len(X_train)} Setups, Test mit {len(X_test)} Setups...")

                model = RandomForestClassifier(
                    n_estimators=1000, 
                    max_depth=None, 
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=1 
                )
                
                # Nur mit dem Trainings-Teil fitten!
                model.fit(X_train, y_train)
                
                # Scores berechnen
                train_score = accuracy_score(y_train, model.predict(X_train))
                test_score = accuracy_score(y_test, model.predict(X_test))
                
                log.info(f"🎯 Train-Score (Memory): {train_score:.2%}")
                log.info(f"💎 ECHTE TEST-GENAUIGKEIT: {test_score:.2%}")

                # Das Modell wird am Ende mit allen Daten finalisiert, bevor es gespeichert wird
                model.fit(X_data, y_data)
                
                save_path = f"ai_models/{symbol}_{tf_name}_model.pkl" # z.B. XAGUSD_M1_model.pkl
                joblib.dump(model, save_path)
                log.info(f"💾 {tf_name} Modell gespeichert.")

if __name__ == "__main__":
    trainer = StrategyAITrainer()
    trainer.train_all()
