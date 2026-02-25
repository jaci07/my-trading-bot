import pandas as pd
import pandas_ta as ta
import time
import json
import os
from datetime import datetime, timedelta
from infrastructure import log

class AdvancedMarketEngine:
    def __init__(self, mt5_connector, db_handler):
        self.mt5 = mt5_connector
        self.db = db_handler
        self.shadow_file = "shadow_trades.json"
        self.active_stats_file = "trade_perf_stats.json"
        
        # Lade laufende Shadow Trades
        self.shadow_trades = self._load_json(self.shadow_file)
        # Lade MFE/MAE Stats für offene Trades
        self.trade_stats = self._load_json(self.active_stats_file)

    def _load_json(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f: return json.load(f)
            except: return [] if "shadow" in filename else {}
        return [] if "shadow" in filename else {}

    def _save_json(self, filename, data):
        with open(filename, "w") as f: json.dump(data, f, indent=4, default=str)

    # ==========================================================
    # 1. MARKT-REGIME (Trend vs. Range) & VELOCITY
    # ==========================================================
    def get_market_regime(self, df):
        """Erkennt: Ist der Markt gerade wild (Trend) oder ruhig (Range)?"""
        try:
            # 1. Check: Haben wir Daten?
            if df is None or df.empty:
                return {"type": "NO_DATA", "adx": 0, "volatility": 0}
            
            # 2. Check: Sind genug Zeilen da? (ADX braucht mind. 14 + Puffer)
            if len(df) < 50:
                # log.warning(f"Zu wenig Kerzen für ADX: {len(df)}")
                return {"type": "NOT_ENOUGH_DATA", "adx": 0, "volatility": 0}

            # 3. Spaltennamen normalisieren (MT5 liefert klein, Pandas TA mag es manchmal anders)
            # Wir erzwingen Kleinschreibung, da pandas_ta das meistens bevorzugt
            df.columns = [c.lower() for c in df.columns]

            # 4. ADX berechnen
            # Wir nutzen try-except direkt hier, falls pandas_ta crasht
            try:
                adx_df = df.ta.adx(high='high', low='low', close='close', length=14)
                
                if adx_df is None or adx_df.empty:
                    curr_adx = 0
                else:
                    # Spalte heißt oft ADX_14 oder ähnlich
                    curr_adx = adx_df.iloc[-1, 0] # Nimm einfach die erste Spalte (ADX)
            except Exception as e:
                log.error(f"ADX Berechnung fehlgeschlagen: {e}")
                curr_adx = 0

            # 5. Bollinger Band Breite (Volatilität)
            try:
                bb = df.ta.bbands(close='close', length=20, std=2)
                # BBU = Upper, BBL = Lower. Namen variieren, wir nehmen Index
                # BBL ist oft Spalte 0, BBM 1, BBU 2
                width = bb.iloc[-1, 2] - bb.iloc[-1, 0] # Upper - Lower
                bb_width = width / df['close'].iloc[-1]
            except:
                bb_width = 0

            # Regime bestimmen
            regime = "RANGING"
            if curr_adx > 25: regime = "TRENDING"
            if curr_adx > 50: regime = "EXTREME_TREND"
            
            return {"type": regime, "adx": round(curr_adx, 2), "volatility": round(bb_width, 4)}

        except Exception as e:
            log.error(f"CRITICAL REGIME ERROR: {e}")
            return {"type": "ERROR", "adx": 0, "volatility": 0}

    def get_tick_velocity(self, symbol):
        """Misst Ticks pro Sekunde (Marktgeschwindigkeit)"""
        try:
            # Hole Ticks der letzten 10 Sekunden
            now = datetime.now()
            start = now - timedelta(seconds=10)
            ticks = self.mt5.mt5.copy_ticks_range(symbol, start, now, self.mt5.mt5.COPY_TICKS_ALL)
            
            if ticks is None: return 0.0
            count = len(ticks)
            return round(count / 10.0, 2) # Ticks/Sekunde
        except: return 0.0

    # ==========================================================
    # UPGRADE: SHADOW TRADING MIT FEATURE-SNAPSHOT
    # ==========================================================
    def spawn_shadow_trades(self, symbol, side, entry_price, current_atr, features):
        """
        Erstellt 5 virtuelle Varianten UND speichert die KI-Features dazu.
        Das ist der Schlüssel, damit die KI später davon lernen kann.
        """
        variants = [
            {"name": "Scalp_Sniper", "sl_m": 1.0, "tp_m": 1.5},
            {"name": "Day_Standard", "sl_m": 1.5, "tp_m": 3.0},
            {"name": "Swing_Runner", "sl_m": 2.5, "tp_m": 6.0},
            {"name": "Tight_Guard",  "sl_m": 0.8, "tp_m": 2.0},
            {"name": "Loose_Risk",   "sl_m": 2.0, "tp_m": 4.0}
        ]
        
        # Wir säubern die Features (keine komplexen Objekte, nur Zahlen)
        clean_features = {k: v for k, v in features.items() if isinstance(v, (int, float, str))}

        for v in variants:
            sl_dist = current_atr * v["sl_m"]
            tp_dist = current_atr * v["tp_m"]
            
            if side == "LONG":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
                
            shadow = {
                "id": f"{symbol}_{int(time.time())}_{v['name']}",
                "symbol": symbol, 
                "side": side,
                "entry": entry_price,
                "sl": sl, 
                "tp": tp,
                "status": "OPEN",
                "start_time": datetime.now().isoformat(),
                "strategy_variant": v["name"],
                "features": clean_features  # <--- HIER IST DAS GOLD!
            }
            self.shadow_trades.append(shadow)
            
        self._save_json(self.shadow_file, self.shadow_trades)
        log.info(f"👻 5 Shadow-Trades (mit Features) für {symbol} gestartet!")

    # ==========================================================
    # UPGRADE: DYNAMISCHE OPTIMIERUNG (Der "Reality Check")
    # ==========================================================
    def analyze_and_optimize(self):
        """
        Analysiert geschlossene Trades und berechnet das perfekte TP/SL Verhältnis.
        Gibt Empfehlungen aus.
        """
        stats_file = "trade_perf_stats.json" # Wir nehmen an, hier landen auch geschlossene (oder wir nutzen DB)
        # HINWEIS: Damit das funktioniert, musst du in update_trade_performance_stats 
        # die geschlossenen Trades in eine "History"-Datei schreiben, statt sie zu löschen.
        
        # Hier die "Pro"-Logik für die Auswertung:
        try:
            # Wir simulieren hier den Zugriff auf die History
            # (Du müsstest im vorherigen Code beim Löschen 'keys_to_delete' in eine history.json speichern)
            history_file = "trade_history_stats.json" 
            if not os.path.exists(history_file): return
            
            with open(history_file, "r") as f:
                trades = json.load(f)
            
            if len(trades) < 10: return # Brauchen Daten
            
            # Analyse
            avg_mfe = sum([t['max_profit_pips'] for t in trades]) / len(trades)
            avg_mae = sum([abs(t['max_drawdown_pips']) for t in trades]) / len(trades)
            
            log.info(f"📊 OPTIMIZER REPORT (Last {len(trades)} Trades):")
            log.info(f"   Ø MFE (Max möglicher Gewinn): {avg_mfe:.1f} Pips")
            log.info(f"   Ø MAE (Max Schmerz): {avg_mae:.1f} Pips")
            
            # Empfehlung berechnen
            suggested_tp = avg_mfe * 0.8 # Wir wollen 80% vom Durchschnitt mitnehmen
            suggested_sl = avg_mae * 1.2 # Wir geben dem Trade 20% mehr Luft als nötig
            
            log.info(f"💡 EMPFEHLUNG: Setze TP auf ca. {suggested_tp:.1f} Pips und SL auf {suggested_sl:.1f} Pips.")
            
            # AUTOMATISCHE ANPASSUNG (Optional - riskant, aber mächtig)
            # self.apply_new_settings(suggested_tp, suggested_sl)

        except Exception as e:
            log.error(f"Optimizer Fehler: {e}")

    def update_shadow_trades(self):
        """Prüft, ob virtuelle Trades gewonnen hätten"""
        active_shadows = [t for t in self.shadow_trades if t["status"] == "OPEN"]
        if not active_shadows: return

        changed = False
        for trade in active_shadows:
            # Live Preis holen
            tick = self.mt5.mt5.symbol_info_tick(trade["symbol"])
            if not tick: continue
            
            bid, ask = tick.bid, tick.ask
            
            outcome = None
            if trade["side"] == "LONG":
                if bid <= trade["sl"]: outcome = "LOSS"
                elif bid >= trade["tp"]: outcome = "WIN"
            else: # SHORT
                if ask >= trade["sl"]: outcome = "LOSS"
                elif ask <= trade["tp"]: outcome = "WIN"
            
            if outcome:
                trade["status"] = outcome
                trade["end_time"] = datetime.now().isoformat()
                log.info(f"👻 SHADOW RESULT: {trade['id']} -> {outcome}")
                changed = True
                
                # OPTIONAL: Hier in DB speichern für KI-Analyse
                # self.db.save_shadow_result(...) 

        if changed:
            self._save_json(self.shadow_file, self.shadow_trades)

    # ==========================================================
    # 3. MFE / MAE TRACKER (Qualitäts-Kontrolle)
    # ==========================================================
    def update_trade_performance_stats(self, positions):
        """Trackt MFE/MAE und archiviert geschlossene Trades."""
        changed = False
        current_ticket_ids = [str(p.ticket) for p in positions]
        
        # 1. UPDATE: Laufende Trades aktualisieren
        for pos in positions:
            ticket = str(pos.ticket)
            symbol = pos.symbol
            entry = pos.price_open
            # Preis holen (Bid für Buy, Ask für Sell um den Exit zu simulieren)
            tick = self.mt5.mt5.symbol_info_tick(symbol)
            if not tick: continue
            current_price = tick.bid if pos.type == 0 else tick.ask
            
            # Init wenn neu
            if ticket not in self.trade_stats:
                self.trade_stats[ticket] = {
                    "symbol": symbol,
                    "max_profit_pips": 0.0,
                    "max_drawdown_pips": 0.0,
                    "entry": entry,
                    "type": "BUY" if pos.type == 0 else "SELL"
                }
            
            stats = self.trade_stats[ticket]
            point = self.mt5.mt5.symbol_info(symbol).point
            
            # Berechne Pips Abstand
            if stats["type"] == "BUY":
                diff = (current_price - entry) / point
            else:
                diff = (entry - current_price) / point
                
            # Update Highscores
            if diff > stats["max_profit_pips"]:
                stats["max_profit_pips"] = diff
                changed = True
            if diff < stats["max_drawdown_pips"]: # drawdown ist negativ
                stats["max_drawdown_pips"] = diff
                changed = True
                
        # 2. CLEANUP & ARCHIV: Geschlossene Trades finden
        keys_to_delete = []
        archive_data = [] # Hier sammeln wir die geschlossenen für die History

        # Welche Tickets sind nicht mehr da?
        for ticket in list(self.trade_stats.keys()):
            if ticket not in current_ticket_ids:
                keys_to_delete.append(ticket)

        # Jetzt verarbeiten wir die geschlossenen
        for k in keys_to_delete:
            stats = self.trade_stats[k] # Daten holen BEVOR wir löschen!
            
            # A) Loggen
            log.info(f"📉 TRADE REPORT {stats['symbol']}: Max Profit: {stats['max_profit_pips']:.1f} Pips | Max DD: {stats['max_drawdown_pips']:.1f} Pips")
            
            # B) Ins Archiv verschieben
            archive_data.append(stats)
            
            # C) Aus dem aktiven Speicher löschen
            del self.trade_stats[k]
            changed = True
        
        # 3. SPEICHERN: Aktive Trades (JSON Update)
        if changed:
            self._save_json(self.active_stats_file, self.trade_stats)

        # 4. SPEICHERN: History (Append an History File)
        if archive_data:
            history_file = "trade_history_stats.json"
            existing = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r") as f: existing = json.load(f)
                except: existing = []
            
            existing.extend(archive_data)
            
            # Nur die letzten 500 Trades behalten (damit die Datei nicht explodiert)
            if len(existing) > 500: existing = existing[-500:]
            
            with open(history_file, "w") as f: json.dump(existing, f, indent=4)

    # ==========================================================
    # 4. SMART ENTRY LOGIC (Die Strategie-Zentrale)
    # ==========================================================
    def check_entry_signal(self, symbol, df, vp_engine):
        """
        SMART ENTRY LOGIC V3.1 (Optimiert mit Volume-Flow & Vacuum-Detection)
        Analysiert Rejections und Breakouts an VAH/VAL.
        Verhindert Trades in klebrigen Konsolidierungen und Fake-Outs ohne Volumen.
        """
        try:
            if df is None or len(df) < 50: return None, None
            
            # 1. Volume Profile Daten holen
            poc, vah, val = vp_engine.calculate_enhanced_profile(df)
            if poc == 0: return None, None

            current_price = df['close'].iloc[-1]
            open_price = df['open'].iloc[-1]
            
            # Hilfsdaten für Momentum und Konsolidierung
            # NEUE BERECHNUNG: Echte durchschnittliche Kerzengröße der letzten 14 Kerzen
            atr = (df['high'].tail(14) - df['low'].tail(14)).mean()
            last_closes = df['close'].tail(5).tolist() 

            # --- OPTIMIERUNG: VOLUMEN-VALIDIERUNG ---
            # Wir suchen die aktive Volumenspalte (tick_volume oder volume)
            vol_col = next((c for c in ['tick_volume', 'volume', 'real_volume'] if c in df.columns), None)
            if vol_col:
                recent_vol = df[vol_col].iloc[-1]
                avg_vol = df[vol_col].tail(20).mean()
                # Ein Ausbruch oder Abpraller ist nur valide, wenn das Volumen > Durchschnitt ist
                vol_confirmed = recent_vol > (avg_vol * 0.85) 
            else:
                vol_confirmed = True # Fallback falls keine Volumendaten

            ''' --- A) STICKY PROTECTION (Gegen Konsolidierung am Level) ---
            touches_vah = sum(1 for p in last_closes if abs(p - vah) < (atr * 0.3))
            touches_val = sum(1 for p in last_closes if abs(p - val) < (atr * 0.3))
            
            if touches_vah >= 3 or touches_val >= 3:
                log.info(f"🛡️ {symbol}: Sticky am Level. Warte auf echten Ausbruch.")
                return None, None
            '''

            # --- A) STICKY PROTECTION ---
            # Prüft, ob der Preis extrem eng am Level festklebt (Chop-Zone)
            last_closes = df['close'].tail(6).tolist()
            
            # Wir verkleinern die "Klebe-Zone" drastisch (von 0.3 auf 0.1 ATR)
            # Nur Kerzen, die FAST EXAKT auf der Linie schließen, zählen als "Sticky"
            touches_vah = sum(1 for p in last_closes if abs(p - vah) <= (atr * 0.1))
            touches_val = sum(1 for p in last_closes if abs(p - val) <= (atr * 0.1))
            
            # Wir fordern mehr Berührungen (4 von 6 statt 3 von 5)
            if touches_vah >= 4 or touches_val >= 4:
                log.info(f"🛡️ {symbol}: Level extrem klebrig. Warte auf klaren Ausbruch.")
                return None, None

            # Kerzen-Struktur für Rejection-Validierung (Dochte)
            upper_wick = df['high'].iloc[-1] - max(open_price, current_price)
            lower_wick = min(open_price, current_price) - df['low'].iloc[-1]
            body = abs(current_price - open_price)

            # We widen the "hit box" to 25% of the ATR. This allows the bot to 
            # register a touch even if it's off by a fraction of a pip (like on TradingView).
            zone = atr * 0.25

            # --- B) STRATEGIE: OBERE KANTE (VAH) ---
            # Did the HIGH of the candle touch the VAH zone? (Matches human eye)
            if df['high'].iloc[-1] >= (vah - zone):
                
                # 1. BREAKOUT (Momentum up)
                # Candle is green and CLOSED above the VAH
                if current_price > open_price and current_price > vah:
                    if vol_confirmed:
                        lva_above = vp_engine.find_nearest_lva(df, current_price, direction="UP")
                        if lva_above is None or (lva_above - current_price) > (atr * 0.8):
                            return "LONG", "Smart_VAH_Breakout_Confirmed"
                
                # 2. REJECTION (Bounce down)
                # Candle tested the VAH, but closed red and BELOW the VAH
                if current_price < open_price and current_price < vah:
                    # Confirmed by a top wick (price pushed up, sellers shoved it back down)
                    if upper_wick > (body * 0.5):
                        return "SHORT", "Smart_VAH_Rejection_Confirmed"

            # --- C) STRATEGIE: UNTERE KANTE (VAL) ---
            # Did the LOW of the candle touch the VAL zone? (Matches human eye)
            elif df['low'].iloc[-1] <= (val + zone):
                
                # 1. BREAKOUT (Momentum down)
                # Candle is red and CLOSED below the VAL
                if current_price < open_price and current_price < val:
                    if vol_confirmed:
                        lva_below = vp_engine.find_nearest_lva(df, current_price, direction="DOWN")
                        if lva_below is None or (current_price - lva_below) > (atr * 0.8):
                            return "SHORT", "Smart_VAL_Breakout_Confirmed"
                
                # 2. REJECTION (Bounce up)
                # Candle tested the VAL, but closed green and ABOVE the VAL
                if current_price > open_price and current_price > val:
                    # Confirmed by a bottom wick (price pushed down, buyers stepped in)
                    if lower_wick > (body * 0.5):
                        return "LONG", "Smart_VAL_Rejection_Confirmed"

            return None, None

        except Exception as e:
            log.error(f"Fehler in check_entry_signal für {symbol}: {e}")
            return None, None
