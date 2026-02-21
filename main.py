# main.py
import warnings
import os

# --- SILENCE THE NOISE ---
# Diese Filter müssen VOR allen anderen Importen stehen!
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# main.py
import time
import sys
from datetime import datetime
import pytz
import pandas as pd
#import yfinance as yf 
import json
from mt5_handler import MT5Handler
from infrastructure import DatabaseHandler, VolumeProfileEngine, AIEngine, log, timedelta
from risk_manager import RiskManager
from settings import cfg
import numpy as np
from advanced_engine import AdvancedMarketEngine # <--- NEU
import joblib
# Unterdrückt die nervigen Parallel-Warnungen
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Optional: Unterdrückt TensorFlow/System Warnungen falls vorhanden
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EnterpriseBot:
    def __init__(self):
        log.info("🚀 INITIALISIERE MT5 SYSTEM...")
        
        # Verbindung zu MT5
        self.mt5 = MT5Handler()
        
        # ==================================================
        # 🛠️ IDENTITÄTS-CHECK (Wichtig für Account-Wechsel)
        # ==================================================
        account_info = self.mt5.mt5.account_info()
        
        if account_info:
            self.current_login = account_info.login 
            log.info(f"🆔 Bot Identität gesetzt: {self.current_login}")
        else:
            self.current_login = 0
            log.warning("⚠️ Konnte Account-ID nicht lesen. Setze auf 0.")

        self.db = DatabaseHandler()
        self.adv_engine = AdvancedMarketEngine(self.mt5, self.db)
        log.info("🧠 Advanced AI Engine geladen (Shadows, MFE/MAE, Regime).")

        self.vp_engine = VolumeProfileEngine()
        self.ai = AIEngine()
        self.risk_manager = RiskManager(self.mt5)
        
        # Hilfsvariablen
        self.data_provider = self 
        self.tz_ny = pytz.timezone('America/New_York')
        self.last_heartbeat = 0
    
    def get_current_features(self, df_m5):
        """Extrahiert die nackten Zahlen, die die AI sieht"""
        df_m5_feat = self.ai.feature_engineering(df_m5.copy())
        if df_m5_feat.empty: return {}
        
        last_row = df_m5_feat.iloc[-1].to_dict()
        clean_features = {k: v for k, v in last_row.items() if isinstance(v, (int, float))}
        return clean_features

    def _close_all_positions(self, comment):
        try:
            positions = self.mt5.mt5.positions_get()
            if positions:
                for pos in positions:
                    # --- DYNAMISCHER FILLING MODE FIX ---
                    symbol_info = self.mt5.mt5.symbol_info(pos.symbol)
                    # Wir prüfen, was der Broker erlaubt (1=FOK, 2=IOC, 3=Beides)
                    filling = symbol_info.filling_mode
                    
                    if filling == 1: # Nur FOK erlaubt
                        fill_type = self.mt5.mt5.ORDER_FILLING_FOK
                    elif filling == 2: # Nur IOC erlaubt
                        fill_type = self.mt5.mt5.ORDER_FILLING_IOC
                    else: # Fallback für alle anderen (meistens RETURN)
                        fill_type = self.mt5.mt5.ORDER_FILLING_RETURN

                    req = {
                        "action": self.mt5.mt5.TRADE_ACTION_DEAL,
                        "position": pos.ticket,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": self.mt5.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.mt5.ORDER_TYPE_BUY,
                        "price": self.mt5.mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else self.mt5.mt5.symbol_info_tick(pos.symbol).ask,
                        "magic": 234000,
                        "comment": comment,
                        "type_time": self.mt5.mt5.ORDER_TIME_GTC,
                        "type_filling": fill_type, # <--- JETZT DYNAMISCH
                    }
                    self.mt5.mt5.order_send(req)
                    time.sleep(0.1)
        except Exception as e:
            log.error(f"Fehler beim Schließen: {e}")

    def learn_from_past_trades(self):
        """
        Vergleicht offene Trades in der DB mit geschlossenen Trades in MT5.
        PRÜFT AUF TICKET-ID, um Verwechslungen zu vermeiden.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, symbol, side, price, features, ticket_id FROM trades WHERE status='OPEN'")
        open_db_trades = cursor.fetchall()
        
        if not open_db_trades: return

        now = datetime.now()
        yesterday = now - timedelta(days=2) 
        history = self.mt5.mt5.history_deals_get(yesterday, now)
        
        if not history: return

        for db_id, symbol, side, entry_price, features_json, db_ticket in open_db_trades:
            for deal in history:
                # Match: Symbol gleich, Entry Out (Exit), Ticket ID gleich
                # deal.position_id ist die ID des Ursprungs-Trades
                is_match = (deal.symbol == symbol) and (deal.entry == 1) and (deal.position_id == db_ticket)
                
                # Fallback für alte Trades ohne Ticket: Schließen ohne lernen
                if not db_ticket:
                     cursor.execute("UPDATE trades SET status='CLOSED' WHERE id=?", (db_id,))
                     self.db.conn.commit()
                     break

                if is_match:
                    profit = deal.profit + deal.swap + deal.commission
                    result_label = 1 if profit > 0 else 0
                    
                    import json
                    try:
                        if features_json:
                            features = json.loads(features_json)
                            self.ai.save_experience(symbol, features, result_label)
                            
                            outcome_str = "WIN 🎉" if profit > 0 else "LOSS 💀"
                            log.info(f"🎓 GELERNT: {symbol} (Ticket {db_ticket}) war ein {outcome_str}. Profit: {profit:.2f}")
                        
                        cursor.execute("UPDATE trades SET status='CLOSED', result=? WHERE id=?", (profit, db_id))
                        self.db.conn.commit()
                        
                    except Exception as e:
                        log.error(f"Lern-Fehler bei {symbol}: {e}")
                    
                    break 

    def is_asset_tradable_now(self, symbol):
        """Prüft Öffnungszeiten pro Asset-Klasse"""
        now = datetime.now(self.tz_ny)
        weekday = now.weekday() # 0=Mo, 6=So
        
        # 1. KRYPTO
        crypto_keywords = ["BTC", "ETH", "LTC", "BCH", "XRP", "DOGE", "SOL"]
        if any(k in symbol for k in crypto_keywords): return True

        # 2. FOREX & INDIZES
        forex_keywords = ["EUR", "USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "XAU", "XAG", "WTI", "BRENT"]
        index_keywords = ["GER40", "US30", "SPX500", "NAS100", "UK100", "JPN225", "AUS200"]
        
        if any(k in symbol for k in forex_keywords + index_keywords):
            if weekday == 5: return False # Samstag zu
            if weekday == 4 and now.hour > 17: return False # Freitag Abend zu
            if weekday == 6 and now.hour < 17: return False # Sonntag früh zu
            return True

        # 3. US-AKTIEN
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if 0 <= weekday <= 4:
            if market_open <= now <= market_close: return True
                
        return False

    def execute_trade(self, symbol, side, strategy, ai_score):
        try:
            # 1. SL/TP Berechnung (Beispielwerte, falls nicht im Signal)
            bid, ask = self.mt5.get_live_price(symbol)
            price = ask if side == "LONG" else bid
            
            # Hier nutzt du deinen RiskManager für die Lot-Größe
            sl_dist = price * 0.002 # 0.2% Puffer
            sl = price - sl_dist if side == "LONG" else price + sl_dist
            tp = price + (sl_dist * 2) if side == "LONG" else price - (sl_dist * 2)
            
            shares = self.risk_manager.calculate_position_size(symbol, price, sl)
            
            if shares > 0:
                success = self.mt5.submit_order(symbol, side, shares, sl, tp, strategy)
                if success:
                    log.info(f"✅ TRADE PLATZIERT: {symbol} {side} | Lots: {shares}")
            else:
                log.warning(f"⚠️ Lot-Größe für {symbol} ist 0. Risiko-Check fehlgeschlagen.")
                
        except Exception as e:
            log.error(f"Fehler in execute_trade: {e}")

    def fetch_candles(self, symbol, timeframe=None):
        """Holt historische Daten DIREKT aus MT5 für den gewünschten Timeframe"""
        # Wenn kein Timeframe angegeben wird, nimm automatisch M5
        if timeframe is None:
            timeframe = self.mt5.mt5.TIMEFRAME_M5 
            
        rates = self.mt5.mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
        
        if rates is None or len(rates) == 0: return None
        
        df_m5 = pd.DataFrame(rates)
        df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
        df_m5.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df_m5.set_index('time', inplace=True)
        return df_m5

    def manage_running_trades(self):
        """
        Verwaltet offene Trades.
        NEU: Night Guard und korrigiertes, dynamisches Smart Trailing.
        """
        positions = self.mt5.mt5.positions_get()
        if not positions: return

        # --- NIGHT GUARD: ZWANGS-SCHLIESSUNG VOR ROLLOVER ---
        now_utc = datetime.utcnow()
        is_rollover_time = (now_utc.hour == 21 and now_utc.minute >= 59) or \
                           (now_utc.hour >= 22) or \
                           (now_utc.hour < 3)
                           
        if is_rollover_time:
            log.warning(f"🌙 NIGHT GUARD: Es ist {now_utc.strftime('%H:%M')} UTC. Schließe alle Positionen vor der Nacht-Pause!")
            for pos in positions:
                request = {
                    "action": self.mt5.mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": self.mt5.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "magic": 234000,
                    "comment": "Night Guard Exit",
                    "type_time": self.mt5.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.mt5.ORDER_FILLING_IOC,
                }
                result = self.mt5.mt5.order_send(request)
                if result.retcode == self.mt5.mt5.TRADE_RETCODE_DONE:
                    log.info(f"✅ {pos.symbol} sicher geschlossen (Spread-Schutz).")
            return 

        # --- SMART TRAILING V2 ---
        try:
            for pos in positions:
                symbol = pos.symbol
                tick = self.mt5.mt5.symbol_info_tick(symbol)
                if not tick: continue
                
                # Preise definieren
                current_price = tick.bid if pos.type == self.mt5.mt5.ORDER_TYPE_BUY else tick.ask
                open_price = pos.price_open
                current_sl = pos.sl
                tp_price = pos.tp

                # Reverse Check
                if self.check_stop_and_reverse(pos, current_price, symbol):
                    continue 
                
                if tp_price == 0: continue

                dist_now = abs(current_price - open_price)
                dist_total = abs(tp_price - open_price)
                if dist_total == 0: continue 
                
                progress = dist_now / dist_total
                
                # FIX: Dynamischer Puffer basierend auf Punktewert (funktioniert bei JPY, EUR, Krypto)
                point = self.mt5.mt5.symbol_info(symbol).point
                BUFFER = point * 30 # 30 Points = exakt 3 Pips Abstand

                # LVA vorbereiten (Nur wenn über 50%, spart CPU)
                lva = None
                if progress >= 0.50:
                    candles = self.mt5.copy_rates_from_pos(symbol, self.mt5.mt5.TIMEFRAME_M5, 0, 500)
                    if candles is not None:
                        df_m5_trail = pd.DataFrame(candles)
                        # FIX: Profil MUSS für dieses Symbol berechnet werden!
                        self.vp_engine.calculate_enhanced_profile(df_m5_trail)
                        direction_lva = "DOWN" if pos.type == self.mt5.mt5.ORDER_TYPE_BUY else "UP"
                        lva = self.vp_engine.find_nearest_lva(df_m5_trail, current_price, direction=direction_lva)

                # ===========================
                # LONG TRADES (pos.type == 0)
                # ===========================
                if pos.type == self.mt5.mt5.ORDER_TYPE_BUY and current_price > open_price:
                    
                    # 1. Break Even
                    if progress >= 0.20 and current_sl < open_price:
                        new_sl = open_price + (point * 10) # 1 Pip Profit sichern
                        self.mt5.modify_position(pos.ticket, new_sl, pos.tp)
                        log.info(f"🛡️ {symbol} LONG: 20% erreicht -> Break Even.")
                        continue

                    # 2. Smart Trailing
                    if progress >= 0.50:
                        lock_pct = 0.30 if progress < 0.70 else 0.55
                        
                        if lva and open_price < lva < current_price:
                            smart_sl = lva - BUFFER
                        else:
                            smart_sl = open_price + (dist_now * lock_pct)

                        # Update wenn neuer SL mind. 2 Pips besser ist
                        if smart_sl > current_sl and (smart_sl - current_sl) > (point * 20):
                            self.mt5.modify_position(pos.ticket, smart_sl, pos.tp)
                            log.info(f"🧱 {symbol} LONG: Smart SL auf {smart_sl:.5f} ({progress*100:.0f}% Fortschritt)")

                # ===========================
                # SHORT TRADES (pos.type == 1)
                # ===========================
                elif pos.type == self.mt5.mt5.ORDER_TYPE_SELL and current_price < open_price:
                    
                    # 1. Break Even
                    if progress >= 0.20 and (current_sl > open_price or current_sl == 0):
                        new_sl = open_price - (point * 10) # 1 Pip Profit sichern
                        self.mt5.modify_position(pos.ticket, new_sl, pos.tp)
                        log.info(f"🛡️ {symbol} SHORT: 20% erreicht -> Break Even.")
                        continue

                    # 2. Smart Trailing
                    if progress >= 0.50:
                        lock_pct = 0.30 if progress < 0.70 else 0.55
                        
                        if lva and current_price < lva < open_price:
                            smart_sl = lva + BUFFER
                        else:
                            # FIX: Minus rechnen bei Short!
                            smart_sl = open_price - (dist_now * lock_pct)

                        # Update wenn neuer SL mind. 2 Pips tiefer (besser) ist
                        if (current_sl == 0 or smart_sl < current_sl) and (current_sl == 0 or (current_sl - smart_sl) > (point * 20)):
                            self.mt5.modify_position(pos.ticket, smart_sl, pos.tp)
                            log.info(f"🧱 {symbol} SHORT: Smart SL auf {smart_sl:.5f} ({progress*100:.0f}% Fortschritt)")

        except Exception as e:
            log.error(f"Fehler im Trailing: {e}")

    def check_stop_and_reverse(self, pos, current_price, symbol):
        """
        Prüft, ob ein Trade gedreht werden muss (Stop & Reverse).
        Logik: Wenn SL fast getroffen ist -> Schließen & Gegentrade öffnen.
        """
        # Einstellungen
        REVERSE_TRIGGER_PIPS = 3.0  # Wie viele Pips VOR dem SL drehen wir?
        MULTIPLIER = 1.0            # 1.0 = Gleiche Größe, 1.5 = Verlust rausholen (Martingale light)
        
        # Nur drehen, wenn es noch kein "Reversal-Trade" ist (erkennbar am Kommentar)
        if pos.comment and "REVERSE" in pos.comment:
            return False

        sl_price = pos.sl
        if sl_price == 0: return False

        # --- CHECK: Sind wir nah am SL? ---
        should_reverse = False
        point = self.mt5.mt5.symbol_info(symbol).point
        
        # Long Trade (SL ist unter uns)
        if pos.type == self.mt5.mt5.ORDER_TYPE_BUY:
            dist_to_sl = (current_price - sl_price) / point
            # Wenn wir weniger als 3 Pips vom SL weg sind (und SL unter Preis ist)
            if dist_to_sl <= REVERSE_TRIGGER_PIPS and current_price > sl_price:
                should_reverse = True
                new_side = "SHORT"
                
        # Short Trade (SL ist über uns)
        elif pos.type == self.mt5.mt5.ORDER_TYPE_SELL:
            dist_to_sl = (sl_price - current_price) / point
            if dist_to_sl <= REVERSE_TRIGGER_PIPS and current_price < sl_price:
                should_reverse = True
                new_side = "LONG"

        # --- EXECUTION ---
        if should_reverse:
            log.warning(f"🔄 SWITCH-SIGNAL für {symbol}: Drehe Position auf {new_side}!")
            
            # 1. Alten Trade schließen
            close_req = {
                "action": self.mt5.mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": symbol,
                "volume": pos.volume,
                "type": self.mt5.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.mt5.ORDER_TYPE_BUY,
                "price": self.mt5.mt5.symbol_info_tick(symbol).bid if pos.type == 0 else self.mt5.mt5.symbol_info_tick(symbol).ask,
                "magic": 234000,
                "comment": "Switch Close",
            }
            res = self.mt5.mt5.order_send(close_req)
            
            if res.retcode != self.mt5.mt5.TRADE_RETCODE_DONE:
                log.error(f"Konnte Switch nicht ausführen (Close failed): {res.comment}")
                return False

            # 2. Neuen Trade öffnen (Gegenrichtung)
            # Wir nehmen den gleichen Abstand für TP/SL wie vorher, nur umgedreht
            vol = pos.volume * MULTIPLIER
            
            # SL/TP für den neuen Trade berechnen (Simpel: 20 Pips SL, 40 Pips TP)
            # Besser wäre dynamisch, aber hier als Beispiel fest:
            sl_dist = 0.0020 * current_price # ca 20 Pips
            tp_dist = 0.0040 * current_price # ca 40 Pips
            
            if new_side == "LONG":
                new_sl = current_price - sl_dist
                new_tp = current_price + tp_dist
                order_type = self.mt5.mt5.ORDER_TYPE_BUY
                price_open = self.mt5.mt5.symbol_info_tick(symbol).ask
            else:
                new_sl = current_price + sl_dist
                new_tp = current_price - tp_dist
                order_type = self.mt5.mt5.ORDER_TYPE_SELL
                price_open = self.mt5.mt5.symbol_info_tick(symbol).bid

            req_new = {
                "action": self.mt5.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": vol,
                "type": order_type,
                "price": price_open,
                "sl": new_sl,
                "tp": new_tp,
                "magic": 234000,
                "comment": "REVERSE Entry", # WICHTIG: Damit wir nicht nochmal drehen!
                "type_time": self.mt5.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.mt5.ORDER_FILLING_IOC,
            }
            
            self.mt5.mt5.order_send(req_new)
            log.info(f"✅ REVERSE SUCCESS: {symbol} jetzt {new_side}")
            return True
            
        return False

    # --- HELPER FÜR DISCORD & SNAPSHOT ---
    def load_settings(self):
        try:
            if not os.path.exists("settings.json"): return {}
            with open("settings.json", "r") as f: return json.load(f)
        except: return None

    def update_status(self, new_status):
        try:
            with open("settings.json", "r") as f: data = json.load(f)
            data["status"] = new_status
            with open("settings.json", "w") as f: json.dump(data, f, indent=4)
        except: pass

    def get_daily_snapshot(self, account, force_reset=False):
        """
        Lädt oder erstellt den Start-Kontostand für den HEUTIGEN Tag.
        force_reset=True -> Überschreibt den Startwert (für Reset via Discord).
        """
        filename = "daily_stats.json"
        today_str = datetime.now().strftime("%Y-%m-%d")
        login_str = str(account.login)
        
        data = {}
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f: data = json.load(f)
            except: data = {}

        account_data = data.get(login_str, {})
        saved_date = account_data.get("date", "")
        
        if saved_date != today_str or force_reset:
            reason = "RESET (Discord)" if force_reset else "Neuer Tag"
            log.info(f"📅 {reason}: Setze Start-Balance für {login_str} NEU auf {account.balance:.2f}")
            
            account_data = {
                "date": today_str,
                "name": account.name,
                "start_balance": account.balance,
                "start_equity": account.equity
            }
            data[login_str] = account_data
            
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
                
            return account.balance
        else:
            return account_data["start_balance"]

    # --- DEINE HAUPTSCHLEIFE (MIT REMOTE CONTROL INTEGRIERT) ---
    # --- DEINE HAUPTSCHLEIFE (KORRIGIERT & FINAL) ---
    def run_strategy_loop(self):
        log.info(f"System bereit. Scanne {len(cfg.SYMBOLS)} Assets auf MT5...")

        while True:
            try:
                # ============================================================
                # 0. SETTINGS LADEN & AUTO-RESET (01:00 UHR)
                # ============================================================
                now = datetime.now()
                settings = self.load_settings()
                if not settings: settings = {}

                if now.hour == 1 and settings.get("status") in ["take_profit", "max_loss", "notified_profit", "notified_loss"]:
                    log.info("🕐 01:00 Uhr: Resette Status für neuen Tag...")
                    self.update_status("running")
                    settings["status"] = "running"
                    self.db.reset_daily_trades()

                status = settings.get("status", "running")

                # ============================================================
                # 1. ACCOUNT WECHSEL CHECK (PRIORITÄT #1)
                # ============================================================
                # Das muss VOR dem "Stop"-Check kommen, damit wir flüchten können.
                json_login = settings.get("target_account")

                if json_login and str(self.current_login) != str(json_login):
                    
                    # Wir erlauben Wechsel auch bei "notified_loss" oder "switch_requested"
                    if status in ["switch_requested", "notified_loss", "login_failed_check_json"] or self.current_login == 0:
                        
                        log.info(f"🔄 REMOTE BEFEHL: Wechsle Account {self.current_login} -> {json_login}")
                        
                        # === DEIN PFAD (Hier ggf. anpassen!) ===
                        MY_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe" 
                        # =======================================

                        try:
                            with open("accounts.json", "r") as f: accounts_db = json.load(f)
                        except: accounts_db = {}
                        
                        if json_login in accounts_db:
                            creds = accounts_db[json_login]
                            
                            log.info(f"🚀 Starte direkten Login-Versuch für {json_login}...")
                            
                            # COMBO-MOVE: Init + Login gleichzeitig
                            init_login_success = self.mt5.mt5.initialize(
                                path=MY_MT5_PATH,
                                login=int(json_login),
                                password=creds["password"],
                                server=creds["server"],
                                timeout=10000
                            )
                            
                            if init_login_success:
                                log.info(f"✅ ERFOLG: Verbindung & Login für {json_login} hergestellt!")
                                self.current_login = json_login
                                self.vp_engine = VolumeProfileEngine() 
                                
                                # Alles resetten und starten
                                self.update_status("running")
                                settings["trading_active"] = True
                                settings["status"] = "running"
                                with open("settings.json", "w") as f: json.dump(settings, f, indent=4)
                                
                                acc = self.mt5.get_account()
                                # WICHTIG: Force Reset, damit er nicht mit 0€ rechnet
                                self.get_daily_snapshot(acc, force_reset=True) 
                                
                            else:
                                err = self.mt5.mt5.last_error()
                                log.error(f"❌ Login fehlgeschlagen! Fehler: {err}")
                                if err[0] == -6: # Authorization failed
                                    log.error("Zugangsdaten falsch oder Konto abgelaufen!")
                                time.sleep(5)
                        else:
                            log.error(f"❌ Ziel-Konto {json_login} fehlt in accounts.json")
                        
                        time.sleep(3)
                        continue # Neustart der Schleife mit neuem Account

                # ============================================================
                # 2. STATUS CHECK (PAUSE / STOPP)
                # ============================================================
                if not settings.get("trading_active", True):
                    log.info("💤 Bot ist PAUSIERT durch Discord. Warte...")
                    time.sleep(10)
                    continue

                if status == "reset_requested":
                    log.info("🔄 RESET SIGNAL: Setze Tages-Statistik zurück...")
                    acc = self.mt5.get_account()
                    if acc: self.get_daily_snapshot(acc, force_reset=True)
                    self.update_status("running")
                    time.sleep(2)
                    continue

                if status in ["max_loss", "take_profit", "notified_loss", "notified_profit"]:
                    log.warning(f"🛑 STOPP-MODUS ({status}). Warte auf Reset via Discord...")
                    time.sleep(10)
                    continue

                # ============================================================
                # 3. PROFIT CHECK (MIT BUG-SCHUTZ)
                # ============================================================
                account = self.mt5.get_account()
                gain_pct = 0.0 # Standardwert

                if account:
                    # SCHUTZ: Wenn Equity fast 0 ist (Fehler beim Laden), nichts tun!
                    if account.equity <= 1.0:
                        # log.warning("⚠️ Equity ungültig (<= 1). Überspringe Profit-Check.")
                        pass 
                    else:
                        start_balance_today = self.get_daily_snapshot(account)
                        current_profit_abs = account.equity - start_balance_today
                        
                        if start_balance_today > 0:
                            gain_pct = (current_profit_abs / start_balance_today) * 100

                        # --- A) TAGESZIEL (+1.0%) ---
                        if gain_pct >= 1.0 and False:
                            log.info(f"🎉 TAGESZIEL ERREICHT (+{gain_pct:.2f}%)!")
                            self.update_status("take_profit")
                            self._close_all_positions("TP Close") # Helper Funktion nutzen oder Code hier einfügen
                            continue

                        # --- B) MAX DRAWDOWN (-2.0%) ---
                        if gain_pct <= -2.0 and False:
                            log.warning(f"☠️ MAX DRAWDOWN ERREICHT ({gain_pct:.2f}%)!")
                            self.update_status("max_loss")
                            self._close_all_positions("SL Close")
                            continue

                # ============================================================
                # 4. NORMALER TRADING LOOP
                # ============================================================
                
                # Zeit-Filter
                current_hour = datetime.now().hour
                if current_hour >= 22 or current_hour < 3:
                    log.info(f"😴 Nacht-Modus. Bot schläft...")
                    time.sleep(60)
                    continue 

                # Heartbeat
                if time.time() - self.last_heartbeat > 300:
                    now_ny = datetime.now(self.tz_ny)
                    log.info(f"💓 Bot läuft | NY-Zeit: {now_ny.strftime('%H:%M')} | Equity: {account.equity if account else 0:.2f}")
                    self.last_heartbeat = time.time()

                # Laufende Trades managen & Lernen
                self.manage_running_trades()

                # ==========================================
                # 📊 UPGRADE 3: BACKGROUND TASKS
                # ==========================================
                # 1. Shadow Trades prüfen
                self.adv_engine.update_shadow_trades()
                
                # 2. MFE / MAE Tracker für laufende Trades
                positions = self.mt5.mt5.positions_get()
                if positions:
                    self.adv_engine.update_trade_performance_stats(positions)
                # ==========================================

                self.learn_from_past_trades()

                if not self.risk_manager.check_can_trade():
                    log.warning("⚠️ Risk Manager blockiert Trading.")
                    time.sleep(60)
                    continue
                
                # ============================================================
                # 3. SCANNING LOOP
                # ============================================================
                for symbol in cfg.SYMBOLS:
                    try:
                        # --- 0. PRE-CHECK: DISCORD ---
                        quick_settings = self.load_settings()
                        if quick_settings:
                            if not quick_settings.get("trading_active", True) or \
                               quick_settings.get("status") != "running":
                                log.info("⚡ Discord-Pause aktiv. Breche Scan ab...")
                                break

                        # --- DIAGNOSE START ---
                        # print(f"Prüfe {symbol}...") # (Optional)
                        
                        if not self.is_asset_tradable_now(symbol): 
                            print(f"🛑 {symbol}: Markt ist geschlossen.")
                            continue
                            
                        if self.db.get_minutes_since_last_trade(symbol) < 15: 
                            print(f"⏳ {symbol}: Cooldown läuft noch.")
                            continue
                        
                        tick = self.mt5.mt5.symbol_info_tick(symbol)
                        if not tick or tick.ask == 0: 
                            print(f"❌ {symbol}: MT5 liefert keine Preise (Marktübersicht prüfen!)")
                            continue
                        
                        spread_pips = (tick.ask - tick.bid) / self.mt5.mt5.symbol_info(symbol).point
                        if spread_pips > 20.0: 
                            print(f"📈 {symbol}: Spread zu hoch ({spread_pips:.1f}).")
                            continue 

                        # --- DATEN HOLEN (DUAL-TF) ---
                        df_m5 = self.fetch_candles(symbol, timeframe=self.mt5.mt5.TIMEFRAME_M5)
                        df_m1 = self.fetch_candles(symbol, timeframe=self.mt5.mt5.TIMEFRAME_M1)
                        
                        if df_m5 is None or df_m5.empty or df_m1 is None or df_m1.empty:
                            print(f"📉 {symbol}: Keine Kerzendaten.")
                            continue

                        bid, ask = self.mt5.get_live_price(symbol)
                        mid_price = (bid + ask) / 2 if bid else 0

                        # --- 1. TECHNISCHE STRATEGIE (M5) ---
                        direction, strategy_name = self.adv_engine.check_entry_signal(symbol, df_m5, self.vp_engine)
                        strat_display = strategy_name if direction else "Wartend (Kein VAH/VAL Break)"

                        # --- 2. KI BEFRAGEN FÜR LOGGING (M5) ---
                        ai_m5 = self.ai.get_ai_prediction(symbol, df_m5, tf_name="M5")
                        
                        best_prob = max(ai_m5['long'], ai_m5['short'], ai_m5['nix'])
                        if best_prob == ai_m5['long']: trend = "LONG"
                        elif best_prob == ai_m5['short']: trend = "SHORT"
                        else: trend = "NIX "

                        print(f"🔎 [{symbol}] Preis:{mid_price:.5f} | AI-Trend ({trend}): {best_prob:.2f} | Strat: {strat_display}")

                        # --- 3. MARKT-FILTER (Velocity) ---
                        velocity = self.adv_engine.get_tick_velocity(symbol)
                        if velocity > 8.0: 
                            continue

                        # --- 4. TECHNISCHES SETUP DA? ---
                        if not direction:
                            continue 
                        
                        # --- 5. KI FÜR M1 BEFRAGEN & AUTO-TRAINING ---
                        ai_m1 = self.ai.get_ai_prediction(symbol, df_m1, tf_name="M1")
                        
                        # Auto-Training (Wie von dir gewünscht)
                        if ai_m5['long'] == 0.0 and ai_m5['short'] == 0.0 and ai_m5['nix'] == 1.0: 
                            log.info(f"🧠 [{symbol}] Kein M5-Modell -> Lerne...")
                            self.ai.train_models(symbol, df_m5)
                            ai_m5 = self.ai.get_ai_prediction(symbol, df_m5, tf_name="M5") # Daten neu laden

                        # --- 6. DER SCHUTZ-FILTER (Nix-Tun Check) ---
                        if (ai_m5["nix"] > ai_m5["long"] and ai_m5["nix"] > ai_m5["short"]) or \
                           (ai_m1["nix"] > ai_m1["long"] and ai_m1["nix"] > ai_m1["short"]):
                            continue

                        if direction == "LONG":
                            score_m5, score_m1 = ai_m5['long'], ai_m1['long']
                        else: 
                            score_m5, score_m1 = ai_m5['short'], ai_m1['short']

                        # --- 7. KI-SCHWELLENWERT (Dual-Threshold) ---
                        THRESHOLD_M5, THRESHOLD_M1 = 0.60, 0.60 
                        if score_m5 < THRESHOLD_M5 or score_m1 < THRESHOLD_M1:
                            continue
                        
                        # ==========================================
                        # 🧠 UPGRADE 2: EXPERTEN-FILTER
                        # ==========================================
                        current_rsi = df_m5['RSI'].iloc[-1] if 'RSI' in df_m5 else 50
                        current_mfi = df_m5['MFI'].iloc[-1] if 'MFI' in df_m5 else 50
                        bb_pct = df_m5['BB_Pct'].iloc[-1] if 'BB_Pct' in df_m5 else 0.5
                
                        # 1. ÜBERKAUFT-SCHUTZ (Für LONG Trades) - FIX: Nutzt jetzt 'direction' statt 'signal'
                        if direction == "LONG":
                            if current_rsi > 75:
                                log.info(f"🛑 Filter: RSI zu hoch ({current_rsi:.1f}). Kein Long.")
                                continue
                            if bb_pct > 1.0:
                                log.info(f"🛑 Filter: Preis über Bollinger Band. Warte Rücksetzer.")
                                continue
                            if current_mfi < 40:
                                log.warning(f"🛑 Filter: Kein Volumen-Support (MFI {current_mfi:.1f}).")
                                continue

                        # 2. ÜBERVERKAUFT-SCHUTZ (Für SHORT Trades)
                        elif direction == "SHORT":
                            if current_rsi < 25:
                                log.info(f"🛑 Filter: RSI zu tief ({current_rsi:.1f}). Kein Short.")
                                continue
                            if bb_pct < 0.0:
                                log.info(f"🛑 Filter: Preis unter Bollinger Band. Warte Pullback.")
                                continue
                            if current_mfi > 60:
                                log.warning(f"🛑 Filter: Zuviel Kaufdruck im Volumen (MFI {current_mfi:.1f}).")
                                continue

                        # 3. DOJI-SCHUTZ (Unsicherheit)
                        if 'Is_Doji' in df_m5 and df_m5['Is_Doji'].iloc[-1] == 1:
                            log.info("🛑 Filter: Letzte Kerze war ein Doji (Unsicherheit). Kein Trade.")
                            continue

                        # WENN WIR HIER SIND: Alle Filter bestanden! ✅

                        # --- SMART ANCHOR & ATR LOGIK ---
                        try:
                            current_atr = df_m5.ta.atr(length=14).iloc[-1]
                        except:
                            current_atr = (df_m5['high'] - df_m5['low']).tail(14).mean()

                        anchor_idx = self.vp_engine.find_last_pivot(df_m5)
                        df_m5_anchored = df_m5.loc[anchor_idx:]
                        if len(df_m5_anchored) < 10: df_m5_anchored = df_m5.tail(96)

                        poc, vah, val = self.vp_engine.calculate_enhanced_profile(df_m5_anchored)
                        vwap = self.vp_engine.calculate_vwap(df_m5)
                        zone_tolerance = current_atr * 0.5

                        log.info(f"🔎 [{symbol}] Filter bestanden | M5-AI:{score_m5:.2f} | M1-AI:{score_m1:.2f} | POC:{poc:.2f}")

                        signal = None
                        
                        # Widerstände für TP/SL finden
                        swing_high_major = df_m5['high'].iloc[-50:].max()
                        swing_low_major = df_m5['low'].iloc[-50:].min()
                        lva_below = self.vp_engine.find_nearest_lva(df_m5, mid_price, direction="DOWN")
                        lva_above = self.vp_engine.find_nearest_lva(df_m5, mid_price, direction="UP")

                        # INLINE FUNKTIONEN (Wie von dir gewünscht)
                        def get_smart_sl(side, entry, lva, swing):
                            MAX_SL_DIST = entry * 0.0035 
                            candidate_sl = swing 
                            use_lva = (side=="LONG" and lva and lva<entry) or (side=="SHORT" and lva and lva>entry)
                            if use_lva: candidate_sl = lva
                            
                            dist = abs(entry - candidate_sl)
                            if dist > MAX_SL_DIST:
                                if side == "LONG": candidate_sl = entry - MAX_SL_DIST
                                else: candidate_sl = entry + MAX_SL_DIST
                            return candidate_sl

                        def get_logical_tp(side, entry, sl):
                            risk = abs(entry - sl)
                            if risk == 0: return entry + (entry*0.001)
                            candidates = []
                            
                            if side == "LONG":
                                if swing_high_major > entry: candidates.append(swing_high_major)
                                if vah > entry: candidates.append(vah)
                                if poc > entry: candidates.append(poc)
                                candidates.append(entry + (risk * 2.0))
                                candidates.sort() 
                            else: 
                                if swing_low_major < entry: candidates.append(swing_low_major)
                                if val < entry: candidates.append(val)
                                if poc < entry: candidates.append(poc)
                                candidates.append(entry - (risk * 2.0))
                                candidates.sort(reverse=True) 

                            best_tp = None
                            for target in candidates:
                                reward = abs(target - entry)
                                rrr = reward / risk
                                if 1 <= rrr <= 2.5: # MIN_RRR und MAX_RRR direkt hier
                                    best_tp = target
                                    break 
                            
                            if best_tp is None:
                                if side == "LONG": best_tp = entry + (risk * 2.0)
                                else: best_tp = entry - (risk * 2.0)
                            return best_tp

                        # --- SETUP SUCHE ---
                        recent_close = df_m5['close'].iloc[-1]

                        # 1. SETUP: VAH Breakout
                        if recent_close > (vah + zone_tolerance) and recent_close > vwap:
                            if not self.db.has_traded_today(symbol, "VAH_Break"):
                                sl_price = vah - zone_tolerance
                                final_sl = get_smart_sl("LONG", mid_price, lva_below, sl_price)
                                if final_sl:
                                    final_tp = get_logical_tp("LONG", mid_price, final_sl)
                                    signal = {"side": "LONG", "tp": final_tp, "sl": final_sl, "setup": "VAH_Break_Smart"}

                        # 2. SETUP: VAL Rejection
                        elif (val - zone_tolerance) < df_m5['low'].iloc[-1] < (val + zone_tolerance) and recent_close > val:
                            if not self.db.has_traded_today(symbol, "VAL_Rej"):
                                 sl_price = df_m5['low'].iloc[-1] - zone_tolerance
                                 final_sl = get_smart_sl("LONG", mid_price, lva_below, sl_price)
                                 if final_sl:
                                     final_tp = get_logical_tp("LONG", mid_price, final_sl)
                                     signal = {"side": "LONG", "tp": final_tp, "sl": final_sl, "setup": "VAL_Rej_Smart"}

                        # 3. SETUP: VAH Rejection (Short)
                        elif (vah - zone_tolerance) < df_m5['high'].iloc[-1] < (vah + zone_tolerance) and recent_close < vah:
                            if not self.db.has_traded_today(symbol, "VAH_Rej"):
                                sl_price = df_m5['high'].iloc[-1] + zone_tolerance
                                final_sl = get_smart_sl("SHORT", mid_price, lva_above, sl_price)
                                if final_sl:
                                    final_tp = get_logical_tp("SHORT", mid_price, final_sl)
                                    signal = {"side": "SHORT", "tp": final_tp, "sl": final_sl, "setup": "VAH_Rej_Smart"}

                        # 4. SETUP: POC Bounce
                        elif abs(mid_price - poc) < zone_tolerance:
                            if df_m5['low'].iloc[-1] <= poc and recent_close > poc and mid_price > vwap:
                                if not self.db.has_traded_today(symbol, "POC_Bounce_Long"):
                                    final_sl = get_smart_sl("LONG", mid_price, lva_below, poc - zone_tolerance)
                                    if final_sl:
                                        final_tp = get_logical_tp("LONG", mid_price, final_sl)
                                        signal = {"side": "LONG", "tp": final_tp, "sl": final_sl, "setup": "POC_Bounce_Smart"}
                            
                            elif df_m5['high'].iloc[-1] >= poc and recent_close < poc and mid_price < vwap:
                                if not self.db.has_traded_today(symbol, "POC_Bounce_Short"):
                                    final_sl = get_smart_sl("SHORT", mid_price, lva_above, poc + zone_tolerance)
                                    if final_sl:
                                        final_tp = get_logical_tp("SHORT", mid_price, final_sl)
                                        signal = {"side": "SHORT", "tp": final_tp, "sl": final_sl, "setup": "POC_Bounce_Smart"}

                        # --- EXECUTION ---
                        if signal:
                            shares = 0 
                            valid_sl = False
                            if signal['side'] == "LONG" and signal['sl'] < mid_price: valid_sl = True
                            if signal['side'] == "SHORT" and signal['sl'] > mid_price: valid_sl = True
                            
                            if not valid_sl: continue

                            profit_potential = abs(signal['tp'] - mid_price)
                            min_profit = mid_price * 0.0015 
                            if profit_potential < min_profit: valid_sl = False

                            if valid_sl:
                                risk_dist = abs(mid_price - signal['sl'])
                                rrr = profit_potential / risk_dist if risk_dist > 0 else 0

                                log.info(f"🚀 SIGNAL: {symbol} {signal['side']} | RRR: {rrr:.2f} | TP: {signal['tp']:.5f}")
                                
                                shares = self.risk_manager.calculate_position_size(symbol, mid_price, signal['sl'])
                            else:
                                log.warning(f"⚠️ {symbol}: Ungültiger SL oder zu wenig Profit. Übersprungen.") 

                            if shares > 0:
                                avg_score = (score_m5 + score_m1) / 2
                                log.info(f"🔥 DUAL-VOLLTREFFER: {symbol} | {signal['setup']} | KI-Score: {avg_score:.2%}")
                                
                                success = self.mt5.submit_order(symbol, signal['side'], shares, signal['sl'], signal['tp'], signal['setup'])
                                    
                                if success:
                                    # 👻 SHADOW TRADES STARTEN
                                    try: current_atr = df_m5.ta.atr(length=14).iloc[-1]
                                    except: current_atr = mid_price * 0.002

                                    current_features = self.ai.feature_engineering(df_m5).iloc[-1].to_dict()    
                                    self.adv_engine.spawn_shadow_trades(symbol, signal['side'], mid_price, current_atr, current_features)

                                    features = self.get_current_features(df_m5)
                                        
                                    ticket_id = 0
                                    try:
                                        time.sleep(0.5) 
                                        open_positions = self.mt5.mt5.positions_get(symbol=symbol)
                                        if open_positions:
                                            newest_pos = sorted(open_positions, key=lambda x: x.ticket)[-1]
                                            ticket_id = newest_pos.ticket
                                    except Exception as e:
                                        log.warning(f"Konnte Ticket-ID für {symbol} nicht sofort finden: {e}")

                                    self.db.log_trade(symbol, signal['side'], shares, mid_price, signal['setup'], features, ticket_id)
                    
                    except Exception as inner_error:
                        log.error(f"❌ Fehler bei {symbol}: {inner_error}")
                        continue 
                        
                # ============================================================
                # 4. LIVE MONITORING (Für das Discord Dashboard)
                # ============================================================
                try:
                    positions = self.mt5.mt5.positions_get()
                    open_trades_count = len(positions) if positions else 0
                    
                    acc = self.mt5.get_account()
                    # Fallback falls 'gain_pct' aus dem oberen Teil der Datei nicht greifbar ist
                    current_gain = gain_pct if 'gain_pct' in locals() else 0.0 
                    
                    monitor_data = {
                        "equity": acc.equity,
                        "balance": acc.balance,
                        "profit_today_pct": current_gain,
                        "open_trades": open_trades_count,
                        "last_update": datetime.now().strftime("%H:%M:%S"),
                        "symbol_active": "Scan beendet..."
                    }
                    
                    with open("monitor.json", "w") as f:
                        json.dump(monitor_data, f)
                except Exception as mon_err:
                    pass 

                time.sleep(5) 

            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                log.error(f"Main Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = EnterpriseBot()
    bot.run_strategy_loop()
