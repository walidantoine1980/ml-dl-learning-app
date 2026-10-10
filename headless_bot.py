import time
import json
import os
import requests
import schedule
from datetime import datetime
import pandas as pd
import yfinance as yf

# Importer les composants du moteur quantitatif existant
from xgboost_trader import (
    MLTrader, MAJOR_STOCKS, add_features, 
    get_macro_data, load_portfolio, save_portfolio
)

BOT_CONFIG_FILE = "bot_config.json"

def load_config():
    """Charge la configuration du bot ou génère les paramètres par défaut."""
    if os.path.exists(BOT_CONFIG_FILE):
        try:
            with open(BOT_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur de lecture {BOT_CONFIG_FILE}: {e}")
    return {
        "webhook_discord": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "tickers": ["Apple Inc. (US)"],
        "run_time": "22:00",
        "is_active": False
    }

def send_alert(config, message):
    """Transmet les alertes via Discord et Telegram selon la configuration."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ALERTE: {message}")
    
    # 1. Discord Webhook
    if config.get("webhook_discord"):
        try:
            requests.post(
                config["webhook_discord"],
                json={"content": f"🤖 **XGBoost Quant Bot**\n{message}"},
                timeout=10
            )
        except Exception as e:
            print(f"Erreur Discord: {e}")
            
    # 2. Telegram Bot API
    if config.get("telegram_bot_token") and config.get("telegram_chat_id"):
        try:
            url = f"https://api.telegram.org/bot{config['telegram_bot_token']}/sendMessage"
            payload = {
                "chat_id": config["telegram_chat_id"],
                "text": f"🤖 *XGBoost Quant Bot*\n{message}",
                "parse_mode": "Markdown"
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"Erreur Telegram: {e}")

def execute_local_trade(ticker, qty, action, current_price):
    """Exécute un ordre d'achat ou de vente dans le portefeuille virtuel local (portfolio.json)."""
    pf = load_portfolio()
    exec_price = current_price * (1.0010 if action.upper() == "BUY" else 0.9990) # Slippage 0.10%
    cost = qty * exec_price
    commission = cost * 0.0005 # Commission 0.05%
    
    if action.upper() == "BUY":
        total_cost = cost + commission
        if pf.get('cash', 0.0) >= total_cost:
            pf['cash'] -= total_cost
            if ticker in pf['positions']:
                old_qty = pf['positions'][ticker]['qty']
                old_price = pf['positions'][ticker]['avg_price']
                new_qty = old_qty + qty
                new_price = ((old_qty * old_price) + total_cost) / new_qty
                pf['positions'][ticker] = {'qty': new_qty, 'avg_price': new_price}
            else:
                pf['positions'][ticker] = {'qty': qty, 'avg_price': exec_price}
                
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY',
                'ticker': ticker,
                'qty': qty,
                'price': exec_price,
                'total': total_cost
            })
            save_portfolio(pf)
            return True, f"Achat virtuel de {qty} actions {ticker} à {exec_price:.2f} $ validé."
        else:
            return False, "Liquidités virtuelles insuffisantes."
    else:
        # Vente
        if ticker in pf['positions'] and pf['positions'][ticker]['qty'] > 0:
            pos_qty = pf['positions'][ticker]['qty']
            qty_to_sell = min(qty, pos_qty)
            net_revenue = (qty_to_sell * exec_price) * (1.0 - 0.0005)
            pf['cash'] += net_revenue
            
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'SELL',
                'ticker': ticker,
                'qty': qty_to_sell,
                'price': exec_price,
                'total': net_revenue,
                'pnl': net_revenue - (qty_to_sell * pf['positions'][ticker]['avg_price'])
            })
            if qty_to_sell >= pos_qty:
                del pf['positions'][ticker]
            else:
                pf['positions'][ticker]['qty'] -= qty_to_sell
            save_portfolio(pf)
            return True, f"Vente virtuelle de {qty_to_sell} actions {ticker} exécutée."
        else:
            return False, "Aucune position ouverte sur ce titre."

def run_trading_job():
    """Exécute l'entraînement nocturne, les prédictions et la gestion de portefeuille."""
    config = load_config()
    if not config.get("is_active", False):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot en veille. Session ignorée.")
        return

    send_alert(config, "🔄 Début de la session quantitative et d'analyse nocturne...")
    
    tickers = config.get("tickers", [])
    if not tickers:
        send_alert(config, "⚠️ Aucune action sélectionnée dans la configuration du bot.")
        return
        
    for ticker in tickers:
        try:
            # Résolution propre du symbole Yahoo Finance
            symbol = MAJOR_STOCKS.get(ticker)
            if not symbol:
                symbol = ticker.split("[")[-1].replace("]", "").strip() if "[" in ticker else ticker

            print(f"\n📊 Analyse de {ticker} ({symbol})...")
            
            # 1. Téléchargement de l'historique
            df_raw = yf.download(symbol, period="2y", interval="1d", progress=False)
            if df_raw is None or df_raw.empty or len(df_raw) < 60:
                print(f"Données insuffisantes pour {symbol}")
                continue
                
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.droplevel(1)
            df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
            
            # 2. Enrichissement technique et macro-économique
            df = add_features(df_raw)
            macro_df = get_macro_data(period="2y", interval="1d")
            if macro_df is not None:
                df = df.join(macro_df, how='left').ffill()
            df = df.dropna()
            
            if df.empty or len(df) < 50:
                continue

            # 3. Entraînement de l'Ensemble IA (XGBoost + LightGBM + RF)
            trader = MLTrader()
            features = trader.train(df, optimize=False, use_wfa=False)
            
            # 4. Prédiction du jour (Clôture pour séance suivante)
            last_row = df.iloc[-1:]
            prob_buy = trader.predict(last_row, features)
            
            if prob_buy is None:
                print(f"Prédiction impossible pour {symbol}")
                continue
                
            prob_pct = prob_buy * 100.0
            var_95 = trader.advanced_metrics.get("VaR (95%)", 0.0) * 100.0
            
            current_price = float(last_row['Close'].values[0])
            action_msg = "Aucun ordre émis (zone de neutralité ou conviction insuffisante)."
            signal_type = "⚪ NEUTRE"
            
            # 5. Logique d'Exécution & Gestion Monétaire Virtuelle
            if prob_buy > 0.55 and prob_pct >= 60.0:
                signal_type = "🟢 SIGNAL D'ACHAT"
                pf = load_portfolio()
                # Allouer 2% du capital disponible par position
                allocated_cash = pf.get('cash', 0.0) * 0.02
                qty = int(allocated_cash / current_price) if current_price > 0 else 0
                if qty > 0:
                    success, t_msg = execute_local_trade(symbol, qty, "BUY", current_price)
                    action_msg = f"Achat de {qty} actions : {t_msg}"
                else:
                    action_msg = "Achat ignoré : liquidités insuffisantes pour 1 action."
                    
            elif prob_buy < 0.45 and (100.0 - prob_pct) >= 60.0:
                signal_type = "🔴 SIGNAL DE VENTE / SHORT"
                pf = load_portfolio()
                if symbol in pf.get('positions', {}) and pf['positions'][symbol].get('qty', 0) > 0:
                    qty = pf['positions'][symbol]['qty']
                    success, t_msg = execute_local_trade(symbol, qty, "SELL", current_price)
                    action_msg = f"Vente/Liquidation de {qty} actions : {t_msg}"
                else:
                    action_msg = "Vente ignorée : aucune position ouverte sur ce titre."
                    
            # 6. Alerte informative
            msg = (
                f"**{ticker}** (`{symbol}`)\n"
                f"📊 {signal_type}\n"
                f"🎯 Probabilité : **{max(prob_pct, 100.0-prob_pct):.1f}%**\n"
                f"🛡️ Value at Risk (VaR 95%) : **{var_95:.2f}%**\n"
                f"💵 Cours clôture : **{current_price:.2f} $**\n"
                f"⚙️ {action_msg}"
            )
            
            # Notification transmise si signal fort
            if (prob_buy > 0.55 and prob_pct >= 60.0) or (prob_buy < 0.45 and (100.0 - prob_pct) >= 60.0):
                send_alert(config, msg)
                
            time.sleep(2) # Respect des limites d'appels API
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {ticker}: {e}")

    send_alert(config, "✅ Session d'entraînement terminée. Modèles mis à jour et en veille.")

def start_scheduler():
    """Superviseur d'exécution planifiée en continu."""
    config = load_config()
    current_run_time = config.get("run_time", "22:00")
    
    print(f"🚀 Démon Headless démarré. Planification fixée à {current_run_time} quotidiennement.")
    schedule.every().day.at(current_run_time).do(run_trading_job)
    send_alert(config, f"🚀 Démon Headless en veille. Planification quotidienne à {current_run_time}.")

    while True:
        try:
            new_config = load_config()
            if not new_config.get("is_active", False):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Arrêt du démon demandé via la configuration.")
                break
                
            # Gestion dynamique des changements d'horaire de passage
            new_run_time = new_config.get("run_time", "22:00")
            if new_run_time != current_run_time:
                print(f"Mise à jour de l'horaire : {current_run_time} -> {new_run_time}")
                schedule.clear()
                current_run_time = new_run_time
                schedule.every().day.at(current_run_time).do(run_trading_job)
                
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("Arrêt manuel du démon.")
            break
        except Exception as e:
            print(f"Erreur superviseur : {e}")
            time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
