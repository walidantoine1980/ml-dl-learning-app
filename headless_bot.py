"""
===============================================================================
DÉMON QUANTITATIF AUTONOME (HEADLESS BOT DAEMON)
XGBoost Stock Predictor Pro - Clean Architecture
===============================================================================
Ce script s'exécute en arrière-plan sans aucune dépendance vers Streamlit.
Fonctionnalités :
- Surveillance proactive des Stops ATR (Stop-Loss & Take-Profit) sur les positions ouvertes
- Téléchargement post-clôture de l'historique et enrichissement technique
- Entraînement et prédiction par l'Ensemble Machine Learning (XGBoost + LightGBM + RF)
- Dimensionnement du risque et exécution virtuelle réaliste (Paper Trading)
- Transmission des alertes enrichies via Discord Webhook et Telegram Bot API
- Supporte le mode planifié (scheduler) ou le mode immédiat (--once)
===============================================================================
"""

import sys
import os
import time
import json
import argparse
import requests
import schedule
from datetime import datetime
import pandas as pd
import yfinance as yf
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import direct du moteur quantitatif indépendant
from quant_core import (
    MLTrader, add_features, get_macro_data, join_macro_data,
    load_portfolio, save_portfolio, execute_equity_trade,
    convert_google_to_yahoo_ticker, get_company_name_from_yahoo
)
from tickers_db import MAJOR_STOCKS

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
        "is_active": True
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


def check_open_positions_risk(config):
    """
    Surveillance proactive des positions ouvertes :
    Déclenche automatiquement les ventes si le Stop-Loss ou le Take-Profit est touché.
    """
    pf = load_portfolio()
    positions = pf.get('positions', {})
    if not positions:
        return
        
    print("\n🛡️ Audit des positions ouvertes en portefeuille...")
    symbols_to_close = []
    
    for symbol, pos in list(positions.items()):
        try:
            ticker_data = yf.Ticker(symbol)
            fast_info = getattr(ticker_data, 'fast_info', None)
            current_price = fast_info.last_price if fast_info and hasattr(fast_info, 'last_price') else None
            
            if current_price is None:
                df = ticker_data.history(period="2d")
                if not df.empty:
                    current_price = float(df['Close'].iloc[-1])
                    
            if current_price is None or current_price <= 0:
                continue
                
            qty = pos.get('qty', 0)
            avg_price = pos.get('avg_price', current_price)
            sl_price = pos.get('stop_loss')
            tp_price = pos.get('take_profit')
            
            hit_reason = None
            if sl_price and current_price <= sl_price:
                hit_reason = f"🛑 STOP-LOSS ATTEINT ({current_price:.2f} $ <= {sl_price:.2f} $)"
            elif tp_price and current_price >= tp_price:
                hit_reason = f"🎯 TAKE-PROFIT ATTEINT ({current_price:.2f} $ >= {tp_price:.2f} $)"
                
            if hit_reason:
                success, t_msg = execute_equity_trade(pf, symbol, qty, "SELL", current_price)
                if success:
                    save_portfolio(pf)
                    pnl_realized = (current_price - avg_price) * qty
                    pnl_pct = (current_price - avg_price) / avg_price if avg_price > 0 else 0.0
                    msg = (
                        f"⚠️ **Sortie Automatique de Position sur {symbol}**\n"
                        f"Motif : {hit_reason}\n"
                        f"Prix d'exécution : **{current_price:.2f} $** (Achat initial: {avg_price:.2f} $)\n"
                        f"PnL Réalisé : **{pnl_realized:+.2f} $ ({pnl_pct:+.2%})**"
                    )
                    send_alert(config, msg)
        except Exception as e:
            print(f"Erreur lors de la vérification de la position {symbol}: {e}")


def run_trading_job():
    """Exécute l'analyse nocturne, les prédictions et la gestion de portefeuille."""
    config = load_config()
    if not config.get("is_active", False):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bot en veille. Session ignorée.")
        return

    send_alert(config, "🔄 Début de la session quantitative et d'analyse de marché...")
    
    # 1. Vérification des risques sur les positions ouvertes
    check_open_positions_risk(config)
    
    tickers = config.get("tickers", [])
    if not tickers:
        send_alert(config, "⚠️ Aucune action sélectionnée dans la configuration du bot.")
        return
        
    macro_df = get_macro_data(period="2y", interval="1d")
    
    for ticker in tickers:
        try:
            # Résolution propre du symbole Yahoo Finance
            symbol = MAJOR_STOCKS.get(ticker)
            if not symbol:
                symbol = ticker.split("[")[-1].replace("]", "").strip() if "[" in ticker else ticker
            yahoo_symbol = convert_google_to_yahoo_ticker(symbol)

            print(f"\n📊 Analyse de {ticker} (`{yahoo_symbol}`)...")
            
            # 1. Téléchargement de l'historique
            df_raw = yf.download(yahoo_symbol, period="2y", interval="1d", progress=False)
            if df_raw is None or df_raw.empty or len(df_raw) < 50:
                print(f"Données insuffisantes pour {yahoo_symbol}")
                continue
                
            if hasattr(df_raw.columns, "droplevel") and df_raw.columns.nlevels > 1:
                df_raw.columns = df_raw.columns.droplevel(1)
            df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
            
            # 2. Enrichissement technique et macro-économique avec conservation de la dernière bougie
            df = add_features(df_raw)
            if macro_df is not None:
                df = join_macro_data(df, macro_df)
                
            if df.empty or len(df) < 40:
                continue

            # 3. Entraînement de l'Ensemble IA (XGBoost + LightGBM + RF)
            model_file = f"models/{yahoo_symbol}_model.pkl"
            trader = None
            if os.path.exists(model_file):
                try:
                    trader = MLTrader.load(model_file)
                except Exception:
                    trader = None
                    
            if trader is None:
                trader = MLTrader()
                features = trader.train(df, optimize=False, use_wfa=False)
                try:
                    trader.save(model_file)
                except Exception:
                    pass
            else:
                features = trader.get_feature_list(df)
            
            # 4. Prédiction pour la séance suivante sur la dernière bougie t disponible
            last_row = df.iloc[-1:]
            prob_buy = trader.predict(last_row, features)
            
            if prob_buy is None:
                print(f"Prédiction impossible pour {yahoo_symbol}")
                continue
                
            prob_pct = prob_buy * 100.0
            var_95 = trader.advanced_metrics.get("VaR (95%)", 0.0) * 100.0
            
            current_price = float(last_row['Close'].iloc[0])
            current_atr = float(last_row['ATR'].iloc[0]) if 'ATR' in last_row.columns and not pd.isna(last_row['ATR'].iloc[0]) else (current_price * 0.02)
            
            action_msg = "Aucun ordre émis (zone de neutralité ou conviction insuffisante)."
            signal_type = "⚪ NEUTRE"
            
            pf = load_portfolio()
            
            # 5. Logique d'Exécution & Gestion Monétaire Virtuelle
            if prob_buy > 0.55 and prob_pct >= 60.0:
                signal_type = "🟢 SIGNAL D'ACHAT"
                # Allouer 2% du capital disponible par position
                allocated_cash = pf.get('cash', 0.0) * 0.02
                qty = int(allocated_cash / current_price) if current_price > 0 else 0
                if qty > 0:
                    success, t_msg = execute_equity_trade(pf, yahoo_symbol, qty, "BUY", current_price, current_atr=current_atr)
                    if success:
                        save_portfolio(pf)
                    action_msg = f"Achat de {qty} actions : {t_msg}"
                else:
                    action_msg = "Achat ignoré : liquidités insuffisantes pour 1 action."
                    
            elif prob_buy < 0.45 and (100.0 - prob_pct) >= 60.0:
                signal_type = "🔴 SIGNAL DE VENTE / LIQUIDATION"
                if yahoo_symbol in pf.get('positions', {}) and pf['positions'][yahoo_symbol].get('qty', 0) > 0:
                    qty = pf['positions'][yahoo_symbol]['qty']
                    success, t_msg = execute_equity_trade(pf, yahoo_symbol, qty, "SELL", current_price)
                    if success:
                        save_portfolio(pf)
                    action_msg = f"Vente de {qty} actions : {t_msg}"
                else:
                    action_msg = "Vente ignorée : aucune position ouverte sur ce titre."
                    
            # 6. Alerte enrichie avec Stop-Loss et Take-Profit calculés
            sl_level = current_price - (1.5 * current_atr)
            tp_level = current_price + (3.0 * current_atr)
            
            msg = (
                f"**{ticker}** (`{yahoo_symbol}`)\n"
                f"📊 Statut : **{signal_type}**\n"
                f"🎯 Probabilité : **{max(prob_pct, 100.0-prob_pct):.1f}%**\n"
                f"🛡️ Value at Risk (VaR 95%) : **{var_95:.2f}%**\n"
                f"💵 Cours de clôture : **{current_price:.2f} $**\n"
                f"🛑 Stop-Loss suggéré : **{sl_level:.2f} $** (1.5 ATR)\n"
                f"🎯 Take-Profit suggéré : **{tp_level:.2f} $** (3.0 ATR)\n"
                f"⚙️ Action : {action_msg}"
            )
            
            # Notification transmise en cas de signal actif
            if (prob_buy > 0.55 and prob_pct >= 60.0) or (prob_buy < 0.45 and (100.0 - prob_pct) >= 60.0):
                send_alert(config, msg)
                
            time.sleep(1) # Respect du rate limit API
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {ticker}: {e}")

    send_alert(config, "✅ Session d'analyse nocturne terminée. Portefeuille et modèles mis à jour.")


def start_scheduler():
    """Superviseur d'exécution planifiée en continu."""
    config = load_config()
    current_run_time = config.get("run_time", "22:00")
    
    print(f"🚀 Démon Headless démarré. Planification quotidienne à {current_run_time}.")
    schedule.every().day.at(current_run_time).do(run_trading_job)
    send_alert(config, f"🚀 Démon Headless actif. Planification quotidienne fixée à {current_run_time}.")

    while True:
        try:
            new_config = load_config()
            if not new_config.get("is_active", False):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Arrêt du démon demandé via la configuration.")
                break
                
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
    parser = argparse.ArgumentParser(description="XGBoost Quant Bot Headless Daemon")
    parser.add_argument("--once", action="store_true", help="Exécute un cycle de trading immédiatement et quitte.")
    args = parser.parse_args()
    
    if args.once:
        print("⚡ Exécution en mode direct (--once)...")
        run_trading_job()
    else:
        start_scheduler()
