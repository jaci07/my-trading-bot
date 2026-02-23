import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# --- 1. EINSTELLUNGEN ---
SYMBOL = "EURUSD"  
THRESHOLD = 0.63    # KI Sicherheit
RRR = 1.5           # Risk-Reward-Ratio 
SWING_PERIOD = 20   # Wie viele Kerzen zurück für den Smart SL?
# ------------------------

def run_visualizer():
    if not mt5.initialize():
        print("❌ MT5 Initialisierung fehlgeschlagen")
        return

    print(f"📊 Lade Daten für {SYMBOL}...")
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from infrastructure import AIEngine
        ai = AIEngine()
    except ImportError:
        print("❌ Konnte AIEngine nicht laden.")
        return

    model_m5_path = f"ai_models/{SYMBOL}_M5_model.pkl"
    model_m1_path = f"ai_models/{SYMBOL}_M1_model.pkl"

    if not os.path.exists(model_m5_path) or not os.path.exists(model_m1_path):
        print("❌ Modelle fehlen!")
        return

    print("🧠 Lade KI-Modelle...")
    model_m5 = joblib.load(model_m5_path)
    model_m1 = joblib.load(model_m1_path)
    model_m5.n_jobs = 1
    model_m1.n_jobs = 1

    print("⏳ Hole Kerzendaten von MT5...")
    rates_m5 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 800)
    rates_m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 4000)

    df_m5 = pd.DataFrame(rates_m5)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
    df_m1 = pd.DataFrame(rates_m1)
    df_m1['time'] = pd.to_datetime(df_m1['time'], unit='s')

    # Broker Point-Wert holen (für Break-Even + 1 Pip)
    symbol_info = mt5.symbol_info(SYMBOL)
    point = symbol_info.point if symbol_info else 0.00001

    print("⚙️ Feature Engineering & Prediction...")
    df_m5_features = ai.feature_engineering(df_m5.copy())
    df_m1_features = ai.feature_engineering(df_m1.copy())

    # Alle unendlichen Werte durch NaN ersetzen und dann alle NaNs mit 0 füllen
    df_m5_features = df_m5_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_m1_features = df_m1_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    feats = model_m5.feature_names_in_ if hasattr(model_m5, "feature_names_in_") else [
             'rsi', 'stoch_k', 'cci', 'rsi_prev1', 'rsi_prev2', 'macd_hist', 'trend_strength', 
             'macd_hist_prev1', 'macd_hist_prev2', 'bb_pct', 'bb_width', 'atr', 'mfi', 
             'obv_slope', 'wick_upper', 'wick_lower', 'is_doji', 'engulfing']

    # --- DER FIX: Klassen intelligent zuordnen ---
    def get_probs(model, features_df):
        probs = model.predict_proba(features_df[feats])
        classes = list(model.classes_) # Die echten Klassen (z.B. [0, 1, 2])
        
        # Finde heraus, in welcher Spalte Long (1) und Short (2) stecken
        idx_long = classes.index(1) if 1 in classes else -1
        idx_short = classes.index(2) if 2 in classes else -1
        
        prob_long = probs[:, idx_long] if idx_long != -1 else np.zeros(len(features_df))
        prob_short = probs[:, idx_short] if idx_short != -1 else np.zeros(len(features_df))
        
        return prob_long, prob_short

    df_m5['long_m5'], df_m5['short_m5'] = get_probs(model_m5, df_m5_features)
    df_m1['long_m1'], df_m1['short_m1'] = get_probs(model_m1, df_m1_features)
    # ---------------------------------------------

    df_merged = pd.merge(df_m5, df_m1[['time', 'long_m1', 'short_m1']], on='time', how='left')
    df_merged['long_m1'] = df_merged['long_m1'].fillna(0)
    df_merged['short_m1'] = df_merged['short_m1'].fillna(0)

    df_merged = df_merged.reset_index(drop=True)


    # --- NEU: Indikatoren für Experten-Filter aus df_m5_features holen ---
    df_merged['rsi'] = df_m5_features['rsi']
    df_merged['bb_pct'] = df_m5_features['bb_pct']
    df_merged['is_doji'] = df_m5_features['is_doji']

    df_merged['mfi'] = df_m5_features['mfi'] if 'mfi' in df_m5_features.columns else 50
    # --- DER FIX: Experten-Filter aus main.py ---
    
    # 1. Basis KI-Signale (Dual-Threshold)
    ki_long = (df_merged['long_m5'] > THRESHOLD) & (df_merged['long_m1'] > THRESHOLD)
    ki_short = (df_merged['short_m5'] > THRESHOLD) & (df_merged['short_m1'] > THRESHOLD)

    # 2. ÜBERKAUFT-SCHUTZ & Volumen-Filter (LONG)
    long_filter = (
        (df_merged['rsi'] <= 75) &      # Kein Long wenn stark überkauft
        (df_merged['bb_pct'] <= 1.0) &  # Preis nicht über dem oberen Bollinger Band
        (df_merged['mfi'] >= 40) &      # Ausreichend Kaufdruck im Volumen
        (df_merged['is_doji'] == 0)     # Keine Unsicherheitskerze
    )

    # 3. ÜBERVERKAUFT-SCHUTZ & Volumen-Filter (SHORT)
    short_filter = (
        (df_merged['rsi'] >= 25) &      # Kein Short wenn stark überverkauft
        (df_merged['bb_pct'] >= 0.0) &  # Preis nicht unter dem unteren Bollinger Band
        (df_merged['mfi'] <= 60) &      # Nicht zu viel Kaufdruck im Volumen
        (df_merged['is_doji'] == 0)     # Keine Unsicherheitskerze
    )

    # Finale Signale: KI-Wahrscheinlichkeit UND alle Filter müssen passen
    df_merged['Dual_Long'] = ki_long & long_filter
    df_merged['Dual_Short'] = ki_short & short_filter
    
    # Fallback für MFI, falls er im Feature-Set anders heißt
    df_merged['mfi'] = df_m5_features['mfi'] if 'mfi' in df_m5_features.columns else 50

    print("⏱️ Simuliere Trades in der Zukunft (Backtest mit Smart SL)...")
    
    wins_long, losses_long, be_long = [], [], []
    wins_short, losses_short, be_short = [], [], []

    for idx, row in df_merged[df_merged['Dual_Long'] | df_merged['Dual_Short']].iterrows():
        if idx < SWING_PERIOD: continue 
        
        entry_price = row['close']
        is_long = row['Dual_Long']
        
        past_data = df_merged.iloc[idx - SWING_PERIOD : idx]
        
        if is_long:
            current_sl = past_data['low'].min()
            if current_sl >= entry_price: current_sl = entry_price - (point * 50) # Fallback, falls SL zu hoch
            risk = entry_price - current_sl
            tp = entry_price + (risk * RRR)
        else:
            current_sl = past_data['high'].max()
            if current_sl <= entry_price: current_sl = entry_price + (point * 50) # Fallback, falls SL zu tief
            risk = current_sl - entry_price
            tp = entry_price - (risk * RRR)
            
        future_data = df_merged.iloc[idx + 1 : idx + 60]
        outcome = "OPEN" # Falls weder TP noch SL getroffen wird
        
        for f_idx, f_row in future_data.iterrows():
            if is_long:
                # 1. Ausstieg prüfen (SL getroffen?)
                if f_row['low'] <= current_sl:
                    if current_sl > entry_price + (point * 5): outcome = "TRAIL_WIN"
                    elif current_sl >= entry_price: outcome = "BE"
                    else: outcome = "LOSS"
                    break
                
                # 2. Ausstieg prüfen (TP getroffen?)
                if f_row['high'] >= tp:
                    outcome = "WIN"
                    break
                    
                # 3. Smart Trailing berechnen (aus deiner main.py)
                current_dist = f_row['high'] - entry_price
                total_dist = tp - entry_price
                if total_dist > 0:
                    progress = current_dist / total_dist
                    
                    if progress >= 0.20 and current_sl < entry_price:
                        current_sl = entry_price + (point * 10) # 1 Pip Profit sichern
                        
                    if progress >= 0.50:
                        lock_pct = 0.30 if progress < 0.70 else 0.55
                        smart_sl = entry_price + (current_dist * lock_pct)
                        # Update wenn neuer SL mind. 20 Punkte besser ist
                        if smart_sl > current_sl and (smart_sl - current_sl) > (point * 20):
                            current_sl = smart_sl

            else: # SHORT LOGIK
                # 1. Ausstieg prüfen (SL getroffen?)
                if f_row['high'] >= current_sl:
                    if current_sl < entry_price - (point * 5): outcome = "TRAIL_WIN"
                    elif current_sl <= entry_price: outcome = "BE"
                    else: outcome = "LOSS"
                    break
                
                # 2. Ausstieg prüfen (TP getroffen?)
                if f_row['low'] <= tp:
                    outcome = "WIN"
                    break
                    
                # 3. Smart Trailing berechnen (aus deiner main.py)
                current_dist = entry_price - f_row['low']
                total_dist = entry_price - tp
                if total_dist > 0:
                    progress = current_dist / total_dist
                    
                    if progress >= 0.20 and current_sl > entry_price:
                        current_sl = entry_price - (point * 10) # 1 Pip Profit sichern
                        
                    if progress >= 0.50:
                        lock_pct = 0.30 if progress < 0.70 else 0.55
                        smart_sl = entry_price - (current_dist * lock_pct)
                        # Update wenn neuer SL mind. 20 Punkte besser ist
                        if smart_sl < current_sl and (current_sl - smart_sl) > (point * 20):
                            current_sl = smart_sl
        
        trade_info = {'time': row['time'], 'price': entry_price}
        if outcome in ["WIN", "TRAIL_WIN"]:
            if is_long: wins_long.append(trade_info)
            else: wins_short.append(trade_info)
        elif outcome == "BE":
            if is_long: be_long.append(trade_info)
            else: be_short.append(trade_info)
        else:
            if is_long: losses_long.append(trade_info)
            else: losses_short.append(trade_info)

    # --- GRAFIK ZEICHNEN ---
    total_trades = len(wins_long) + len(losses_long) + len(wins_short) + len(losses_short) + len(be_long) + len(be_short)
    total_wins = len(wins_long) + len(wins_short)
    total_be = len(be_long) + len(be_short)
    
    # Echte Winrate (Sieger / (Alle Trades - BreakEvens)) -> BE verfälscht die Statistik nicht negativ
    active_trades = total_trades - total_be
    winrate = (total_wins / active_trades * 100) if active_trades > 0 else 0

    print(f"🎨 Chart: {total_wins} Wins | {total_be} Break-Evens | {len(losses_long)+len(losses_short)} Losses")
    print(f"📈 Realistische Winrate: {winrate:.1f}%")
    
    plt.figure(figsize=(16, 8))
    plt.plot(df_merged['time'], df_merged['close'], label=f'{SYMBOL} M5 Preis', color='black', alpha=0.5, linewidth=1)

    if wins_long:
        df_wl = pd.DataFrame(wins_long)
        plt.scatter(df_wl['time'], df_wl['price'], color='lime', label='Gewinner (LONG)', marker='^', s=150, edgecolors='darkgreen', zorder=5)
    
    if wins_short:
        df_ws = pd.DataFrame(wins_short)
        plt.scatter(df_ws['time'], df_ws['price'], color='red', label='Gewinner (SHORT)', marker='v', s=150, edgecolors='darkred', zorder=5)

    if be_long:
        df_be_l = pd.DataFrame(be_long)
        plt.scatter(df_be_l['time'], df_be_l['price'], color='cyan', label='Break-Even (LONG)', marker='.', s=120, edgecolors='blue', zorder=4)

    if be_short:
        df_be_s = pd.DataFrame(be_short)
        plt.scatter(df_be_s['time'], df_be_s['price'], color='cyan', label='Break-Even (SHORT)', marker='.', s=120, edgecolors='blue', zorder=4)

    if losses_long:
        df_ll = pd.DataFrame(losses_long)
        plt.scatter(df_ll['time'], df_ll['price'], color='gray', label='Verlierer (LONG)', marker='^', s=100, edgecolors='black', alpha=0.6, zorder=3)

    if losses_short:
        df_ls = pd.DataFrame(losses_short)
        plt.scatter(df_ls['time'], df_ls['price'], color='gray', label='Verlierer (SHORT)', marker='v', s=100, edgecolors='black', alpha=0.6, zorder=3)

    plt.title(f"{SYMBOL} KI Visual Backtest mit Smart SL | Winrate: {winrate:.1f}% (RRR {RRR})")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visualizer()
