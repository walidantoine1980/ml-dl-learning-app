import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from google import genai
except ImportError:
    genai = None
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import time
import pickle
import os
import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import subprocess
import scipy.stats as si
from scipy.optimize import brentq
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False
from tickers_db import MAJOR_STOCKS, PREDEFINED_PORTFOLIOS
import glob

# Import du moteur quantitatif centralisé
from quant_core import (
    add_features, get_macro_data as qc_get_macro_data, join_macro_data,
    MLTrader, convert_google_to_yahoo_ticker, get_company_name_from_yahoo,
    get_option_multiplier_and_legislation, get_black_litterman_weights,
    black_scholes, find_strike_for_delta, run_options_backtest_simulation,
    get_portfolio_path, load_portfolio, save_portfolio,
    get_options_portfolio_path, load_options_portfolio, save_options_portfolio,
    execute_equity_trade, evaluate_multi_regime_triggers
)
from academy_core import (
    render_indicators_academy,
    render_xgboost_academy,
    render_strategy_academy,
    render_advanced_academy,
    render_options_academy,
    render_news_academy
)
# Le chargement des portefeuilles personnalisés est maintenant géré dynamiquement par tickers_db.py

@st.cache_resource(show_spinner="Chargement du modèle Deep Learning NLP (FinBERT)...")
def load_finbert():
    try:
        import torch
        from transformers import pipeline
        # Configuration sur CPU ou GPU selon la disponibilité
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_macro_data(period, interval):
    """Télécharge l'historique des indicateurs macroéconomiques avec cache Streamlit"""
    return qc_get_macro_data(period=period, interval=interval)

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    """Analyse le sentiment des dernières actualités via Deep Learning (FinBERT)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0, []
            
        finbert = load_finbert()
        if finbert is None:
            return 0, [] # Fallback si le modèle crash
            
        polarities = []
        articles = []
        for n in news[:5]:
            content = n.get('content', {})
            title = content.get('title', '')
            if title:
                result = finbert(title)[0]
                label = result['label']
                score = result['score']
                
                # Traduire le label FinBERT en polarité
                if label == 'positive':
                    pol = score
                elif label == 'negative':
                    pol = -score
                else:
                    pol = 0.0
                    
                polarities.append(pol)
                articles.append({'title': title, 'link': content.get('clickThroughUrl', ''), 'polarity': pol})
                
        avg_polarity = np.mean(polarities) if polarities else 0
        return avg_polarity, articles
    except Exception:
        return 0, []

@st.cache_data(ttl=3600)
def get_implied_volatility(ticker, spot_price=None):
    """Récupère la volatilité implicite moyenne (IV) des options ATM les plus proches."""
    try:
        stock = yf.Ticker(ticker)
        dates = stock.options
        if not dates:
            return 0.20 # Fallback
        
        # Prendre la première échéance (front month)
        chain = stock.option_chain(dates[0])
        
        if spot_price:
            # Filtrer les calls autour de la monnaie (ATM)
            calls = chain.calls
            calls['dist'] = abs(calls['strike'] - spot_price)
            atm_calls = calls.sort_values('dist').head(3)
            iv = atm_calls['impliedVolatility'].mean()
        else:
            iv = chain.calls['impliedVolatility'].median()
            
        if pd.isna(iv) or iv == 0:
            return 0.20
        return float(iv)
    except Exception:
        return 0.20



# Création du dossier pour les modèles
os.makedirs("models", exist_ok=True)

# Configuration de la page
st.set_page_config(page_title="XGBoost Stock Predictor Pro", layout="wide", page_icon="📈")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #3e4259;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .signal-sell {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR QUANTITATIF ---
# Les fonctions de calcul d'indicateurs (add_features), l'Ensemble Machine Learning (MLTrader),
# et les convertisseurs de tickers sont désormais importés directement depuis quant_core.py.

# --- MODES D'AFFICHAGE ---

def run_single_mode(ticker, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None):
    st.subheader(f"Analyse Individuelle : {get_company_name_from_yahoo(ticker)}")
    model_path = f"models/{ticker}_model.pkl"
    
    if f"trader_{ticker}" not in st.session_state:
        st.session_state[f"trader_{ticker}"] = MLTrader()
        
    trader = st.session_state[f"trader_{ticker}"]

    with st.spinner(f"Téléchargement des données pour {ticker}..."):
        df_raw = yf.download(ticker, period=period, interval=interval, progress=False)
        if df_raw.empty:
            st.error(f"Erreur lors du téléchargement. Vérifiez le ticker (recherché: {ticker}).")
            return
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)
        # S'assurer qu'il n'y a pas de colonnes dupliquées (arrive si yfinance télécharge plusieurs tickers)
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
            
        macro_df = get_macro_data(period, interval)

    df = add_features(df_raw)
    if macro_df is not None:
        df = join_macro_data(df, macro_df)
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🧠 Entraîner l'IA sur cette action", use_container_width=True):
            with st.status("Entraînement en cours...") as status:
                try:
                    features = trader.train(df, optimize=optimize_model, use_wfa=use_wfa, wfa_train_window=wfa_train_window, wfa_step=wfa_step, wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date)
                    st.session_state[f"features_{ticker}"] = features
                    status.update(label="✅ Modèle entraîné !", state="complete")
                except ValueError as ve:
                    status.update(label="❌ Erreur de paramétrage WFA", state="error")
                    st.error(f"⚠️ {ve}")
                    return
                except Exception as ex:
                    status.update(label="❌ Erreur d'entraînement", state="error")
                    st.error(f"Erreur imprévue : {ex}")
                    return
                
    with col2:
        if trader.is_trained:
            if st.button("💾 Sauvegarder", use_container_width=True):
                trader.save(model_path)
                st.toast("Modèle sauvegardé sur le disque !", icon="💾")
                
    with col3:
        if os.path.exists(model_path):
            if st.button("📂 Charger", use_container_width=True):
                st.session_state[f"trader_{ticker}"] = MLTrader.load(model_path)
                st.toast("Modèle chargé depuis le disque !", icon="📂")
                st.rerun()
                
    with col4:
        if trader.is_trained:
            st.metric("Précision du Modèle", f"{trader.accuracy:.2%}")

    if trader.is_trained:
        last_row = df.iloc[-1:]
        # Default features if loaded from disk without re-training session
        features = trader.get_feature_list(df)
        prob = trader.predict(last_row, features)
        st.progress(float(prob))
        
        st.divider()
        # --- NLP Sentiment Logic ---
        st.subheader("📰 Filtre de Sentiment Actuel (FinBERT)")
        try:
            avg_sentiment, articles = get_news_sentiment(ticker)
            if articles:
                sentiment_text = "Neutre"
                color = "gray"
                if avg_sentiment > 0.15:
                    sentiment_text = "Positif 🟢"
                    color = "green"
                elif avg_sentiment < -0.15:
                    sentiment_text = "Négatif 🔴"
                    color = "red"
                    
                st.markdown(f"Score de Sentiment FinBERT : **<span style='color:{color}'>{avg_sentiment:+.2f} ({sentiment_text})</span>**", unsafe_allow_html=True)
                
                # Alerte de contradiction
                if prob > 0.55 and avg_sentiment < -0.2:
                    st.warning("⚠️ **Contradiction Fondamentale** : L'IA mathématique recommande l'ACHAT, mais l'IA linguistique détecte des nouvelles très NÉGATIVES. Risque de fausse cassure élevé.")
                elif prob < 0.45 and avg_sentiment > 0.2:
                    st.warning("⚠️ **Contradiction Fondamentale** : L'IA mathématique recommande la VENTE, mais l'IA linguistique détecte des nouvelles très POSITIVES.")
            else:
                st.write("Aucune actualité récente disponible pour le NLP.")
        except:
            st.write("Impossible de récupérer les actualités.")
            
        st.divider()
        st.header("📝 Plan de Trading du Jour (Money Management)")
        
        c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1])
        with c1:
            st.subheader("🔮 Signal")
            current_price = last_row['Close'].values[0]
            sma_200 = last_row['SMA_200'].values[0] if 'SMA_200' in last_row.columns else current_price
            
            # --- FILTRE DE RÉGIME (SMA 200) ---
            regime_bull = current_price > sma_200
            
            if prob == 0.0:
                st.markdown("<span class='signal-sell'>⚠️ KRACH (VIX>30) - CASH</span>", unsafe_allow_html=True)
            elif prob > 0.55:
                if regime_bull:
                    st.markdown("<span class='signal-buy'>🟢 ACHAT FORT</span>", unsafe_allow_html=True)
                    st.write("Contexte : Régime Haussier (>SMA 200)")
                else:
                    st.markdown("<span style='color:orange; font-weight:bold; font-size:20px;'>🟡 ACHAT RISQUÉ</span>", unsafe_allow_html=True)
                    st.write("⚠️ Contre-Tendance (<SMA 200)")
                st.write(f"Confiance IA : {prob:.1%}")
            elif prob < 0.45:
                if not regime_bull:
                    st.markdown("<span class='signal-sell'>🔴 VENTE / SHORT</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='signal-sell'>🔴 VENTE / CASH</span>", unsafe_allow_html=True)
                st.write(f"Confiance Baisse : {1-prob:.1%}")
            else:
                st.write("⚪ **NEUTRE** / Attente")
                
        with c2:
            st.subheader("📰 Sentiment News")
            avg_pol, articles = get_news_sentiment(ticker)
            if avg_pol > 0.1:
                st.success(f"Positif (+{avg_pol:.2f})")
            elif avg_pol < -0.1:
                st.error(f"Négatif ({avg_pol:.2f})")
            else:
                st.info(f"Neutre ({avg_pol:.2f})")
                
        with c3:
            st.subheader("🛡️ Gestion du Risque (Kelly)")
            current_atr = last_row['ATR'].values[0] if 'ATR' in last_row.columns else current_price * 0.02
            
            stop_loss = current_price - (1.5 * current_atr) if prob > 0.5 else current_price + (1.5 * current_atr)
            take_profit = current_price + (3 * current_atr) if prob > 0.5 else current_price - (3 * current_atr)
            risk_per_share = abs(current_price - stop_loss)
            
            # --- CRITÈRE DE KELLY (Half-Kelly) ---
            adv_m = getattr(trader, 'advanced_metrics', {})
            kelly_pct = adv_m.get('Kelly %', adv_m.get('Trade Kelly %', 0.0))
            half_kelly_pct = adv_m.get('Half Kelly %', min(0.05, max(0.0, kelly_pct / 2.0)))
            if kelly_pct > 0:
                risk_pct = min(half_kelly_pct, 0.05) # Max 5% du compte
                st.info(f"Half-Kelly Actif : {risk_pct*100:.2f}% (Plein Kelly: {kelly_pct*100:.2f}%)")
            elif adv_m.get('Total Trades', 0) > 0:
                risk_pct = 0.01
                st.warning("Edge statistique insuffisant (Kelly ≤ 0). Risque prudent par défaut : 1.00%")
            else:
                risk_pct = 0.01
                st.info("Historique de trades insuffisant pour Kelly. Risque par défaut : 1.00%")
                
            max_loss = initial_capital * risk_pct
            
            if prob > 0.55:
                # Régime ACHAT
                stop_loss = current_price - (1.5 * current_atr)
                take_profit = current_price + (3.0 * current_atr)
                risk_per_share = abs(current_price - stop_loss)
                position_size = int(max_loss / risk_per_share) if risk_per_share > 0 else 0
                total_investment = position_size * current_price
                if total_investment > initial_capital:
                    position_size = int(initial_capital / current_price)
                    total_investment = position_size * current_price
                st.markdown(f"""
                - **Orientation** : 🟢 **ACHAT FORT (LONG)**
                - **Prix Actuel** : {current_price:.2f} $
                - **Taille de position** : {position_size} actions ({total_investment:,.2f} $)
                - **Stop-Loss (1.5 ATR)** : <span style='color:red'>{stop_loss:.2f} $</span>
                - **Take-Profit (3.0 ATR)** : <span style='color:green'>{take_profit:.2f} $</span>
                """, unsafe_allow_html=True)
            elif prob < 0.45:
                # Régime VENTE / SHORT
                stop_loss = current_price + (1.5 * current_atr)
                take_profit = current_price - (3.0 * current_atr)
                position_size = 0
                total_investment = 0.0
                st.markdown(f"""
                - **Orientation** : 🔴 **VENTE / CASH (SHORT)**
                - **Prix Actuel** : {current_price:.2f} $
                - **Position suggérée** : **0 action (100% Cash / Put)**
                - **Invalidation Haussière (SL)** : <span style='color:red'>{stop_loss:.2f} $</span>
                - **Cible Baissière (TP)** : <span style='color:green'>{take_profit:.2f} $</span>
                """, unsafe_allow_html=True)
            else:
                # Régime NEUTRE / ATTENTE
                sl_ref = current_price - (1.5 * current_atr)
                tp_ref = current_price + (3.0 * current_atr)
                position_size = 0
                total_investment = 0.0
                stop_loss = sl_ref
                take_profit = tp_ref
                st.markdown(f"""
                - **Orientation** : ⚪ **NEUTRE (ATTENTE)**
                - **Prix Actuel** : {current_price:.2f} $
                - **Position suggérée** : **0 action (En attente de signal clair)**
                - *Niveaux de référence Achat : SL = {sl_ref:.2f} $ | TP = {tp_ref:.2f} $*
                """)
                
        with c4:
            st.subheader("📑 Exécution & Export")
            sig_name = 'ACHAT' if prob > 0.55 else ('VENTE/SHORT' if prob < 0.45 else 'NEUTRE/ATTENTE')
            export_text = f"DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
            export_text += f"ACTION: {ticker}\n"
            export_text += f"SIGNAL: {sig_name}\n"
            export_text += f"PRIX ENTREE: {current_price:.2f} $\n"
            export_text += f"STOP-LOSS: {stop_loss:.2f} $\n"
            export_text += f"TAKE-PROFIT: {take_profit:.2f} $\n"
            export_text += f"TAILLE POSITION: {position_size} actions\n"
            export_text += f"RISQUE MAX: {max_loss:.2f} $\n"
            
            st.download_button(
                label="📥 Télécharger le Plan",
                data=export_text,
                file_name=f"Trading_Plan_{ticker}.txt",
                mime="text/plain"
            )
            
            # --- PAPER TRADING BUTTON ---
            if prob > 0.55 and position_size > 0:
                if st.button("🚀 Exécuter sur le Paper Trading"):
                    pf = load_portfolio()
                    if not isinstance(pf.get('positions'), dict):
                        pf['positions'] = {}
                    if not isinstance(pf.get('history'), list):
                        pf['history'] = []
                    # Simulation Event-Driven (Slippage + Commissions)
                    exec_price = current_price * 1.0010  # +0.10% Slippage
                    cost = position_size * exec_price
                    commission = cost * 0.0005           # +0.05% Frais
                    total_cost = cost + commission
                    
                    if pf.get('cash', 0.0) >= total_cost:
                        pf['cash'] -= total_cost
                        # Update positions
                        if ticker in pf['positions']:
                            old_qty = pf['positions'][ticker]['qty']
                            old_price = pf['positions'][ticker]['avg_price']
                            new_qty = old_qty + position_size
                            new_price = ((old_qty * old_price) + total_cost) / new_qty
                            pf['positions'][ticker] = {'qty': new_qty, 'avg_price': new_price}
                        else:
                            pf['positions'][ticker] = {'qty': position_size, 'avg_price': exec_price}
                        
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY',
                            'ticker': ticker,
                            'qty': position_size,
                            'price': exec_price,
                            'total': total_cost
                        })
                        save_portfolio(pf)
                        st.success(f"Ordre exécuté virtuellement ! {position_size} actions {ticker} achetées (Slippage et Frais inclus).")
                    else:
                        st.error("Fonds insuffisants dans le Paper Trading.")
            elif prob < 0.45:
                if st.button("🔴 Liquider Position (Paper Trading)"):
                    pf = load_portfolio()
                    if not isinstance(pf.get('positions'), dict):
                        pf['positions'] = {}
                    if not isinstance(pf.get('history'), list):
                        pf['history'] = []
                    if ticker in pf['positions'] and pf['positions'][ticker].get('qty', 0) > 0:
                        qty = pf['positions'][ticker]['qty']
                        
                        # Simulation Event-Driven (Slippage + Commissions)
                        exec_price = current_price * 0.9990 # -0.10% Slippage
                        gross_revenue = qty * exec_price
                        commission = gross_revenue * 0.0005 # -0.05% Frais
                        net_revenue = gross_revenue - commission
                        
                        pf['cash'] += net_revenue
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'SELL',
                            'ticker': ticker,
                            'qty': qty,
                            'price': exec_price,
                            'total': net_revenue,
                            'pnl': net_revenue - (qty * pf['positions'][ticker]['avg_price'])
                        })
                        del pf['positions'][ticker]
                        save_portfolio(pf)
                        st.success(f"Position liquidée virtuellement ! {qty} actions {ticker} vendues (Slippage et Frais déduits).")
                    else:
                        st.warning("Aucune position ouverte sur ce titre.")

        # --- NOUVEAU: COCKPIT DES DÉCLENCHEURS INSTITUTIONNELS (TRIGGERS ENGINE) ---
        st.divider()
        st.subheader("🎯 Cockpit des Déclencheurs Institutionnels (Multi-Triggers)")
        current_vix = float(last_row['VIX'].values[0]) if 'VIX' in last_row.columns and not pd.isna(last_row['VIX'].values[0]) else None
        triggers = evaluate_multi_regime_triggers(
            last_row.iloc[0], current_price, current_atr=current_atr, 
            trader_prob=prob, current_vix=current_vix
        )
        
        trig_c1, trig_c2, trig_c3, trig_c4 = st.columns(4)
        with trig_c1:
            st.markdown(f"**Régime Macro**\n\n{triggers['trend_regime']['badge']}")
            st.caption(triggers['trend_regime']['action'])
        with trig_c2:
            st.markdown(f"**Bouclier RSI**\n\n{triggers['rsi_shield']['badge']}")
            st.caption(triggers['rsi_shield']['action'])
        with trig_c3:
            st.markdown(f"**Flux Volume**\n\n{triggers['volume_surge']['badge']}")
            st.caption(triggers['volume_surge']['action'])
        with trig_c4:
            st.markdown(f"**Volatilité & Squeeze**\n\n{triggers['bollinger_squeeze']['badge']}")
            st.caption(triggers['bollinger_squeeze']['action'])
            
        r_levels = triggers.get('risk_levels', {})
        st.info(
            f"🔒 **Déclencheur de Sécurisation (Breakeven)** : Dès que le cours atteint **{r_levels.get('breakeven_trigger', 0.0):.2f} $** (+1.5 ATR / 1R de gain), "
            f"le Stop-Loss initial (**{r_levels.get('initial_sl', 0.0):.2f} $**) est automatiquement remonté à **{current_price:.2f} $** pour un risque nul !"
        )

        # --- NOUVEAU: STRATÉGIE OPTIONS RECOMMANDÉE ---
        st.divider()
        st.subheader("🛡️ Stratégie Dérivée Recommandée (Options)")
        current_price_opt = last_row['Close'].values[0]
        
        # Approximation de la volatilité et du taux sans risque
        vol_opt = df['Returns'].std() * np.sqrt(252)
        r_opt = 0.05
        
        if prob > 0.55:
            multiplier, legislation = get_option_multiplier_and_legislation(ticker)
            # Recommande un Call ATM à 30 jours
            strike_opt = round(current_price_opt, 2)
            t_opt_days = 30
            price_opt, delta_opt, gamma_opt, theta_opt, vega_opt = black_scholes(current_price_opt, strike_opt, t_opt_days/365, r_opt, vol_opt, "call")
            st.success(f"**Recommandation : ACHAT de CALL (Pari Haussier avec Levier)**")
            c_o1, c_o2, c_o3 = st.columns(3)
            c_o1.markdown(f"""
            - **Type** : Call
            - **Strike** : {strike_opt} $
            - **Expiration** : Dans {t_opt_days} jours
            """)
            c_o2.markdown(f"""
            - **Prime Estimée (Coût)** : {price_opt:.2f} $ par action
            - **Coût total du contrat (x{multiplier})** : {price_opt*multiplier:.2f} $
            - **Norme** : {legislation}
            """)
            c_o3.markdown(f"""
            - **Levier Estimé** : {(current_price_opt * delta_opt) / price_opt:.1f}x
            """)
            
            # --- EXÉCUTION CALL OPTIONS ---
            if price_opt > 0:
                cost_per_contract = price_opt * multiplier
                qty_to_buy = int(max_loss / cost_per_contract) if cost_per_contract > 0 else 0
                qty_to_buy = max(1, qty_to_buy) # Au moins 1 contrat
                total_premium = qty_to_buy * cost_per_contract
                
                if st.button(f"🚀 Acheter {qty_to_buy} Contrats CALL (Paper Trading Options)"):
                    pf_opt = load_options_portfolio()
                    if not isinstance(pf_opt.get('positions'), dict):
                        pf_opt['positions'] = {}
                    if not isinstance(pf_opt.get('history'), list):
                        pf_opt['history'] = []
                    if pf_opt.get('cash', 0.0) >= total_premium:
                        pf_opt['cash'] -= total_premium
                        contract_id = f"{ticker}_CALL_{strike_opt}_{t_opt_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        pf_opt['positions'][contract_id] = {
                            'ticker': ticker,
                            'type': 'call',
                            'strike': strike_opt,
                            'days_to_expiry': t_opt_days,
                            'premium': price_opt,
                            'qty': qty_to_buy,
                            'multiplier': multiplier,
                            'underlying_price_at_buy': current_price_opt,
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"CALL Strike {strike_opt}",
                            'qty': qty_to_buy,
                            'premium_paid': price_opt,
                            'total_cost': total_premium
                        })
                        save_options_portfolio(pf_opt)
                        st.success(f"{qty_to_buy} contrats CALL achetés virtuellement ! Retrouvez-les dans 'Paper Trading (Options)'.")
                    else:
                        st.error("Fonds virtuels (Options) insuffisants.")
        elif prob < 0.45:
            multiplier, legislation = get_option_multiplier_and_legislation(ticker)
            # Recommande un Put ATM à 30 jours
            strike_opt = round(current_price_opt, 2)
            t_opt_days = 30
            price_opt, delta_opt, gamma_opt, theta_opt, vega_opt = black_scholes(current_price_opt, strike_opt, t_opt_days/365, r_opt, vol_opt, "put")
            st.error(f"**Recommandation : ACHAT de PUT (Pari Baissier ou Couverture)**")
            c_o1, c_o2, c_o3 = st.columns(3)
            c_o1.markdown(f"""
            - **Type** : Put
            - **Strike** : {strike_opt} $
            - **Expiration** : Dans {t_opt_days} jours
            """)
            c_o2.markdown(f"""
            - **Prime Estimée (Coût)** : {price_opt:.2f} $ par action
            - **Coût total du contrat (x{multiplier})** : {price_opt*multiplier:.2f} $
            - **Norme** : {legislation}
            """)
            c_o3.markdown(f"""
            - **Levier Estimé** : {(current_price_opt * abs(delta_opt)) / price_opt:.1f}x
            """)
            
            # --- EXÉCUTION PUT OPTIONS ---
            if price_opt > 0:
                cost_per_contract = price_opt * multiplier
                qty_to_buy = int(max_loss / cost_per_contract) if cost_per_contract > 0 else 0
                qty_to_buy = max(1, qty_to_buy) # Au moins 1 contrat
                total_premium = qty_to_buy * cost_per_contract
                
                if st.button(f"🔴 Acheter {qty_to_buy} Contrats PUT (Paper Trading Options)"):
                    pf_opt = load_options_portfolio()
                    if not isinstance(pf_opt.get('positions'), dict):
                        pf_opt['positions'] = {}
                    if not isinstance(pf_opt.get('history'), list):
                        pf_opt['history'] = []
                    if pf_opt.get('cash', 0.0) >= total_premium:
                        pf_opt['cash'] -= total_premium
                        contract_id = f"{ticker}_PUT_{strike_opt}_{t_opt_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        pf_opt['positions'][contract_id] = {
                            'ticker': ticker,
                            'type': 'put',
                            'strike': strike_opt,
                            'days_to_expiry': t_opt_days,
                            'premium': price_opt,
                            'qty': qty_to_buy,
                            'multiplier': multiplier,
                            'underlying_price_at_buy': current_price_opt,
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"PUT Strike {strike_opt}",
                            'qty': qty_to_buy,
                            'premium_paid': price_opt,
                            'total_cost': total_premium
                        })
                        save_options_portfolio(pf_opt)
                        st.success(f"{qty_to_buy} contrats PUT achetés virtuellement ! Retrouvez-les dans 'Paper Trading (Options)'.")
                    else:
                        st.error("Fonds virtuels (Options) insuffisants.")
        else:
            st.info("Aucune stratégie d'options recommandée en régime neutre ou très incertain.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 Graphique Technique", "📉 Métriques de Backtest", "🧠 Importance des Variables"])
    
    with tab1:
        st.subheader(f"Analyse Technique : {ticker}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=('Prix & Moyennes', 'RSI'),
                           row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prix'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.5)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.5)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if trader.is_trained:
            bt_df = trader.backtest_results
            bt_df['Port_Strategy'] = initial_capital * bt_df['Cum_Strategy_Return']
            bt_df['Port_Market'] = initial_capital * bt_df['Cum_Market_Return']
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Port_Strategy'], name='Portefeuille XGBoost ($)', line=dict(color='#00ff00', width=2)))
            fig_perf.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Port_Market'], name='Portefeuille Buy & Hold ($)', line=dict(color='gray', dash='dash')))
            fig_perf.update_layout(height=500, template="plotly_dark", title="Évolution du Portefeuille (Données de Test)", yaxis_title="Valeur en $")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.subheader("📉 Larmes Quantitatives (Tearsheet)")
            adv = getattr(trader, 'advanced_metrics', {})
            
            final_strategy = bt_df['Port_Strategy'].iloc[-1]
            final_market = bt_df['Port_Market'].iloc[-1]
            profit_strategy = final_strategy - initial_capital
            profit_market = final_market - initial_capital
            diff_profit = profit_strategy - profit_market
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Capital Final", f"{final_strategy:,.2f} $", delta=f"{profit_strategy:+,.2f} $")
            m2.metric("Win Rate", f"{adv.get('Win Rate', 0)*100:.1f}%")
            m3.metric("Max Drawdown", f"{adv.get('Max Drawdown', 0)*100:.2f}%")
            m4.metric("Sharpe", f"{adv.get('Sharpe Ratio', 0):.2f}")
            if diff_profit > 0:
                m5.metric("Alpha", f"+{diff_profit:,.2f} $", delta="IA bat le marché", delta_color="normal")
            else:
                m5.metric("Alpha", f"{diff_profit:,.2f} $", delta="Marché bat l'IA", delta_color="inverse")
                
            st.divider()
            n1, n2, n3 = st.columns(3)
            var_pct = adv.get('VaR (95%)', 0) * 100
            cvar_pct = adv.get('CVaR', 0) * 100
            prob_ruin = adv.get('Probabilité de Ruine', 0) * 100
            n1.metric("VaR Historique (95%)", f"{var_pct:.2f}%", help="Perte maximale journalière dans 95% des cas.")
            n2.metric("CVaR (Expected Shortfall)", f"{cvar_pct:.2f}%", help="Perte moyenne lorsque la VaR est dépassée (Risque extrême).")
            n3.metric("Probabilité de Ruine (MC)", f"{prob_ruin:.2f}%", help="Probabilité de perdre 20% du capital (Bootstrap Monte Carlo).")
                
            st.divider()
            st.subheader("🎲 Simulation de Monte Carlo (Value at Risk)")
            st.markdown("Projection probabiliste sur 30 jours (1000 scénarios) pour mesurer le risque extrême.")
            
            # Paramètres
            days_to_simulate = 30
            num_simulations = 1000
            
            # Calcul de la dérive (drift) et de la volatilité de la stratégie
            strat_returns = bt_df['Strategy_Return'].dropna()
            mu = strat_returns.mean()
            sigma = strat_returns.std()
            
            # Génération des chemins aléatoires
            last_price = bt_df['Port_Strategy'].iloc[-1]
            simulation_df = pd.DataFrame()
            
            # Numpy vectorization pour plus de rapidité (au lieu d'une boucle lente)
            daily_returns = np.random.normal(mu, sigma, (days_to_simulate, num_simulations))
            price_paths = np.zeros_like(daily_returns)
            price_paths[0] = last_price
            for t in range(1, days_to_simulate):
                price_paths[t] = price_paths[t-1] * (1 + daily_returns[t])
                
            simulation_df = pd.DataFrame(price_paths)
            
            # Affichage graphique (on prend un échantillon de 100 courbes pour ne pas surcharger)
            fig_mc = go.Figure()
            for x in range(100):
                fig_mc.add_trace(go.Scatter(y=simulation_df[x], mode='lines', line=dict(width=1, color='rgba(0, 255, 0, 0.05)')))
            fig_mc.update_layout(height=400, template="plotly_dark", title="100 Scénarios de Monte Carlo", showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Calcul de la VaR (Value at Risk à 95%)
            final_prices = simulation_df.iloc[-1]
            var_95 = np.percentile(final_prices, 5) # Le 5ème centile pire
            perte_potentielle = last_price - var_95
            
            st.error(f"⚠️ **Value at Risk (VaR 95%) sur 1 mois** : Dans 95% des cas, la valeur du portefeuille ne chutera pas en dessous de **{var_95:,.2f} $**.")
            st.info(f"Pire perte estimée sur 30 jours : **-{perte_potentielle:,.2f} $**")
        else:
            st.info("Entraînez le modèle pour voir les performances de backtest.")

    with tab3:
        st.subheader("🤖 Dans le cerveau de XGBoost")
        if trader.is_trained:
            # --- NOUVEAU : SHAP WATERFALL ---
            st.markdown("### 🔬 Explicabilité de la décision du jour (SHAP)")
            if not HAS_SHAP or shap is None:
                st.info("Le package `shap` n'est pas installé dans cet environnement. Exécutez `pip install shap` pour afficher les explications SHAP.")
            else:
                try:
                    # Dans un VotingClassifier, les modèles entraînés sont dans estimators_
                    fitted_xgb = trader.model.estimators_[0]
                    explainer = shap.TreeExplainer(fitted_xgb)
                    shap_values = explainer.shap_values(last_row[features])
                    
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    elif len(shap_values.shape) > 1:
                        sv = shap_values[0]
                    else:
                        sv = shap_values
                    
                    shap_df = pd.DataFrame({'Feature': features, 'Contribution': sv})
                    shap_df = shap_df.sort_values(by='Contribution', key=abs, ascending=True).tail(10)
                    
                    fig_shap = go.Figure(go.Waterfall(
                        name="SHAP", orientation="h",
                        measure=["relative"] * len(shap_df),
                        y=shap_df['Feature'],
                        x=shap_df['Contribution'],
                        text=[f"{x:+.3f}" for x in shap_df['Contribution']],
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": "red"}},
                        increasing={"marker": {"color": "green"}}
                    ))
                    fig_shap.update_layout(
                        title="Impact des variables sur le dernier signal",
                        template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors du calcul SHAP : {e}")

            st.divider()
            st.subheader("Importance Globale des Variables")
            df_imp = trader.feature_importances
            fig_imp = go.Figure(go.Bar(
                x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
                marker=dict(color=df_imp['Importance'], colorscale='viridis')
            ))
            fig_imp.update_layout(title="Importance des Variables", xaxis_title="Poids", yaxis_title="Indicateur", template="plotly_dark", height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info("💡 Pour comprendre comment XGBoost fonctionne, consultez '🤖 Académie: XGBoost' dans le menu de gauche.")
        else:
            st.warning("Entraînez le modèle pour débloquer l'analyse du cerveau de l'IA.")

# get_black_litterman_weights est importé de quant_core.py

def run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None):
    st.subheader("🌐 Mode Portefeuille Institutionnel (Black-Litterman)")
    st.markdown("""
    Ce mode combine **deux piliers de la finance quantitative** :
    1. **Le Prior (L'Équilibre)** : L'allocation initiale est gérée par la Parité des Risques (Inverse Volatilité) pour lisser le risque statistique.
    2. **Les Vues (L'Alpha XGBoost)** : Le modèle de *Black-Litterman* (Goldman Sachs) prend l'équilibre initial et le "tord" mathématiquement en fonction de la conviction de notre IA sur chaque action.
    """)
    
    if st.button("🧠 Entraîner et Optimiser le Portefeuille", use_container_width=True):
        with st.status("Analyse Macro & Entraînement des modèles...") as status:
            data_dict = {}
            returns_dict = {}
            volatility_dict = {}
            
            for t in tickers:
                st.write(f"Récupération pour {t}...")
                df_raw = yf.download(t, period=period, interval=interval, progress=False)
                if not df_raw.empty:
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.droplevel(1)
                    # S'assurer qu'il n'y a pas de colonnes dupliquées (évite les DataFrame dans df_raw['Close'])
                    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
                    
                    data_dict[t] = df_raw
                    ret = df_raw['Close'].pct_change().dropna()
                    returns_dict[t] = ret
                    vol_hist = ret.std()
                    
                    if pd.isna(vol_hist) or float(vol_hist) <= 0:
                        volatility_dict[t] = 0.01
                    else:
                        volatility_dict[t] = float(vol_hist)

            # --- 1. Poids d'Équilibre (Risk Parity) ---
            if not volatility_dict:
                st.error("❌ Aucune donnée de cours valide n'a pu être récupérée pour les tickers sélectionnés.")
                return
            sum_inv_vol = sum(1/v for v in volatility_dict.values())
            if sum_inv_vol == 0:
                sum_inv_vol = 1e-9
            market_weights = {t: (1/v)/sum_inv_vol for t, v in volatility_dict.items()}
            
            # --- 2. Entraînement IA & Génération des Vues ---
            views_dict = {}
            for t, df_raw in data_dict.items():
                st.write(f"Entraînement de l'IA pour {t}...")
                macro_df = get_macro_data(period, interval)
                df = add_features(df_raw)
                if macro_df is not None:
                    df = join_macro_data(df, macro_df)
                trader = MLTrader()
                try:
                    features = trader.train(df, optimize=optimize_model, use_wfa=use_wfa, wfa_train_window=wfa_train_window, wfa_step=wfa_step, wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date)
                except ValueError as ve:
                    st.warning(f"⚠️ {t} : {ve}")
                    # Repli sur entraînement standard
                    features = trader.train(df, optimize=optimize_model, use_wfa=False)
                st.session_state[f"port_trader_{t}"] = trader
                
                # Prédiction actuelle pour le Black-Litterman (La Vue)
                last_row = df.iloc[-1:]
                prob = trader.predict(last_row, features)
                # Transformation de la probabilité en rendement excédentaire attendu
                # Ex: prob=0.60 => (0.60 - 0.50) * 0.50 = +0.05 (+5% d'excès)
                expected_excess = (prob - 0.50) * 0.50 if prob is not None else 0.0
                views_dict[t] = expected_excess
                
                if "probs_dict" not in st.session_state:
                    st.session_state["probs_dict"] = {}
                st.session_state["probs_dict"][t] = prob
                
            # --- 3. Optimisation Black-Litterman ---
            st.write("🧮 Optimisation de la Matrice de Covariance (Black-Litterman)...")
            returns_df = pd.DataFrame(returns_dict).ffill().bfill().fillna(0.0)
            
            # Application du modèle
            bl_weights = get_black_litterman_weights(returns_df, market_weights, views_dict)
            
            st.session_state["bl_weights"] = bl_weights
            st.session_state["market_weights"] = market_weights
            
            for t, w in bl_weights.items():
                st.session_state[f"port_capital_{t}"] = initial_capital * w
                
            status.update(label="✅ Portefeuille Institutionnel généré et optimisé !", state="complete")
            
    valid_tickers = [t for t in tickers if f"port_trader_{t}" in st.session_state]
    is_ready = len(valid_tickers) > 0
    
    if is_ready:
        if "bl_weights" in st.session_state:
            st.subheader("💼 Allocation Optimisée du Portefeuille (Signal d'Investissement)")
            bl_weights = st.session_state["bl_weights"]
            market_weights = st.session_state["market_weights"]
            probs_dict = st.session_state.get("probs_dict", {})
            
            cols = st.columns(min(len(valid_tickers), 4))
            idx = 0
            for t, w in bl_weights.items():
                if t in valid_tickers:
                    m_w = market_weights.get(t, 0)
                    diff = w - m_w
                    prob = probs_dict.get(t)
                    if prob is None: prob = 0.5
                    amount = initial_capital * w
                    
                    with cols[idx % 4]:
                        st.metric(label=get_company_name_from_yahoo(t), value=f"{w*100:.1f}%", delta=f"{diff*100:+.1f}% (Ajustement IA)")
                        st.write(f"Capital: **{amount:,.2f} $**")
                        st.write(f"Probabilité IA: **{prob*100:.1f}%**")
                        
                        if amount > 10: # Ne proposer l'investissement que si le montant est significatif
                            if st.button(f"🛒 Investir {t}", key=f"inv_{t}"):
                                pf = load_portfolio()
                                if not isinstance(pf.get('positions'), dict):
                                    pf['positions'] = {}
                                if not isinstance(pf.get('history'), list):
                                    pf['history'] = []
                                if pf.get('cash', 0.0) < amount:
                                    st.error("Liquidités insuffisantes !")
                                else:
                                    try:
                                        current_price = yf.download(t, period="1d", progress=False)['Close'].iloc[-1]
                                        if isinstance(current_price, pd.Series):
                                            current_price = current_price.iloc[0]
                                        exec_price = float(current_price) * 1.001
                                        qty = amount / exec_price
                                        commission = amount * 0.0005
                                        pf['cash'] -= (amount + commission)
                                        if t in pf['positions']:
                                            old_qty = pf['positions'][t]['qty']
                                            old_avg = pf['positions'][t]['avg_price']
                                            pf['positions'][t]['avg_price'] = ((old_qty * old_avg) + (qty * exec_price)) / (old_qty + qty)
                                            pf['positions'][t]['qty'] += qty
                                        else:
                                            pf['positions'][t] = {'qty': qty, 'avg_price': exec_price}
                                        pf['history'].append({
                                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'action': 'BUY (Action indiv. BL)',
                                            'ticker': t, 'qty': qty, 'price': exec_price, 'total': amount + commission, 'pnl': 0.0
                                        })
                                        save_portfolio(pf)
                                        st.toast(f"✅ {t} ajouté au portefeuille avec succès !", icon="✅")
                                    except Exception as e:
                                        st.error(f"Erreur d'exécution: {e}")
                    idx += 1
            st.divider()

        all_port_strategy = pd.DataFrame()
        all_port_market = pd.DataFrame()
        
        for t in valid_tickers:
            trader = st.session_state[f"port_trader_{t}"]
            # Récupère le capital alloué ou un capital équitable par défaut
            capital_for_t = st.session_state.get(f"port_capital_{t}", initial_capital/len(valid_tickers))
            bt = trader.backtest_results
            
            if bt is None or bt.empty:
                continue
                
            if all_port_strategy.empty:
                all_port_strategy['Total'] = bt['Cum_Strategy_Return'] * capital_for_t
                all_port_market['Total'] = bt['Cum_Market_Return'] * capital_for_t
            else:
                all_port_strategy['Total'] = all_port_strategy['Total'].add(bt['Cum_Strategy_Return'] * capital_for_t, fill_value=0)
                all_port_market['Total'] = all_port_market['Total'].add(bt['Cum_Market_Return'] * capital_for_t, fill_value=0)
                
        if all_port_strategy.empty:
            st.warning("⚠️ Pas assez de données de backtest disponibles pour générer le graphique de synthèse. Certaines actions n'ont peut-être pas assez d'historique.")
        else:
            all_port_strategy = all_port_strategy.dropna()
            all_port_market = all_port_market.dropna()
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=all_port_strategy.index, y=all_port_strategy['Total'], name='Portefeuille Quantitatif ($)', line=dict(color='#00ff00', width=2)))
            fig_perf.add_trace(go.Scatter(x=all_port_market.index, y=all_port_market['Total'], name='Portefeuille Buy & Hold ($)', line=dict(color='gray', dash='dash')))
            fig_perf.update_layout(height=500, template="plotly_dark", title="Évolution Globale du Portefeuille (Risk Parity)", yaxis_title="Valeur en $")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            final_strategy = all_port_strategy['Total'].iloc[-1]
            final_market = all_port_market['Total'].iloc[-1]
            profit_strategy = final_strategy - initial_capital
            profit_market = final_market - initial_capital
            diff_profit = profit_strategy - profit_market
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Capital Final (IA)", f"{final_strategy:,.2f} $", delta=f"{profit_strategy:+,.2f} $")
            m2.metric("Capital Final (Marché)", f"{final_market:,.2f} $", delta=f"{profit_market:+,.2f} $", delta_color="off")
            
            if diff_profit > 0:
                m3.metric("Surperformance Globale", f"+{diff_profit:,.2f} $", delta="L'IA bat le marché", delta_color="normal")
            else:
                m3.metric("Sous-performance Globale", f"{diff_profit:,.2f} $", delta="Le marché bat l'IA", delta_color="inverse")
    else:
        st.warning("Veuillez cliquer sur Entraîner pour générer la synthèse globale.")


# --- PAGES ACADÉMIE ---

def page_indicators():
    render_indicators_academy()

def page_xgboost():
    render_xgboost_academy()

def page_news():
    render_news_academy()

def format_large_number(num):
    if num is None or pd.isna(num): return "N/A"
    try:
        num = float(num)
    except Exception:
        return str(num)
    sign = "-" if num < 0 else ""
    abs_num = abs(num)
    if abs_num >= 1e12: return f"{sign}{abs_num/1e12:.2f} T"
    if abs_num >= 1e9: return f"{sign}{abs_num/1e9:.2f} B"
    if abs_num >= 1e6: return f"{sign}{abs_num/1e6:.2f} M"
    return f"{sign}{abs_num:,.2f}"

def page_fundamentals(tickers):
    st.title("🏢 Analyse Fondamentale & Screener")
    st.markdown("Plongez dans les bilans financiers et comparez la santé réelle des entreprises sélectionnées.")
    
    with st.expander("📖 Guide Détaillé des Indicateurs Fondamentaux (Lexique)"):
        st.markdown("""
        **📈 Valorisation & Prix**
        *   **Capitalisation Boursière :** La valeur totale de l'entreprise sur les marchés financiers. (Prix d'une action × Nombre total d'actions).
        *   **P/E Ratio (Price-to-Earnings ou PER) :** Le "Prix sur Bénéfices". Il indique combien d'années de bénéfices il faut pour rembourser le prix de l'action. Un PER élevé indique de fortes attentes de croissance.
        *   **PEG Ratio :** Le P/E Ratio divisé par la croissance attendue des bénéfices. Un PEG < 1 signale souvent que l'action est sous-évaluée par rapport à sa croissance future.
        *   **Price / Sales :** Capitalisation divisée par le Chiffre d'Affaires. Très utile pour les entreprises en forte croissance qui ne dégagent pas encore de bénéfices nets.
        *   **EV / EBITDA :** Valeur d'Entreprise sur EBITDA (bénéfices bruts). C'est le ratio préféré des fonds d'investissement (M&A, Private Equity) car il prend en compte la dette.

        **💵 Trésorerie & Dette**
        *   **Debt / Equity (Dette sur Capitaux Propres) :** Mesure le niveau d'endettement. Un ratio > 100 signifie que l'entreprise a plus de dettes que d'argent apporté par les actionnaires (Risque élevé en cas de crise).
        *   **Current Ratio :** Liquidité à court terme (Actifs / Dettes à moins d'un an). S'il est < 1, l'entreprise pourrait avoir du mal à payer ses factures immédiates.
        *   **Free Cash Flow (Flux de Trésorerie Disponible) :** L'argent "réel" qu'il reste en caisse après avoir payé le fonctionnement et les investissements. C'est le cash qui sert à payer les dividendes ou racheter des actions.
        *   **Rendement Dividende :** Le pourcentage du prix de l'action qui est payé chaque année en cash aux actionnaires.

        **🚀 Rentabilité & Risque**
        *   **Marge Nette :** Le pourcentage de profit pur réalisé sur chaque vente. Une entreprise avec une marge élevée possède un fort pouvoir de fixation des prix (Pricing Power).
        *   **ROE (Return On Equity) :** La rentabilité financière. Si vous donnez 100$ à la direction, combien de profit génère-t-elle ? Un ROE > 15% est généralement excellent.
        *   **Beta :** L'indicateur de volatilité systémique. Beta = 1 équivaut au marché. Beta > 1 signifie que l'action amplifie les mouvements du marché (Plus risquée). Beta < 1 indique une valeur défensive.
        """)

    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral (Configuration Globale).")
        return
        
    comparative_data = []
    detailed_views = []
    
    with st.spinner("Téléchargement des données fondamentales en cours..."):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if 'shortName' not in info and 'longName' not in info:
                    continue
                    
                name = info.get('longName', info.get('shortName', ticker))
                
                # Extraction des métriques pour la table comparative
                comparative_data.append({
                    "Entreprise": name,
                    "Ticker": ticker,
                    "Secteur": info.get('sector', 'N/A'),
                    "Cap. Boursière": info.get('marketCap'),
                    "P/E (Actuel)": info.get('trailingPE'),
                    "P/E (Futur)": info.get('forwardPE'),
                    "PEG Ratio": info.get('pegRatio'),
                    "Marge Nette": info.get('profitMargins'),
                    "ROE": info.get('returnOnEquity'),
                    "Debt/Equity": info.get('debtToEquity'),
                    "Dividende": info.get('dividendYield'),
                    "Beta": info.get('beta')
                })
                
                # Sauvegarde l'info complète pour l'affichage détaillé
                detailed_views.append({'ticker': ticker, 'name': name, 'info': info})
                
            except Exception as e:
                st.error(f"Erreur lors de la récupération pour {ticker}: {e}")

    # --- 1. TABLEAU COMPARATIF ---
    if len(comparative_data) > 1:
        st.header("📊 Synthèse Comparative (Screener)")
        df_comp = pd.DataFrame(comparative_data)
        
        # Formatage du DataFrame pour l'affichage
        df_display = df_comp.copy()
        df_display['Cap. Boursière'] = df_display['Cap. Boursière'].apply(lambda x: format_large_number(x))
        df_display['Marge Nette'] = df_display['Marge Nette'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        df_display['ROE'] = df_display['ROE'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        df_display['Dividende'] = df_display['Dividende'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        
        # Arrondir les colonnes numériques
        for col in ['P/E (Actuel)', 'P/E (Futur)', 'PEG Ratio', 'Debt/Equity', 'Beta']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.divider()

    # --- 2. FICHES DÉTAILLÉES ---
    st.header("🔍 Fiches Détaillées (Deep Dive)")
    for item in detailed_views:
        info = item['info']
        name = item['name']
        ticker = item['ticker']
        
        st.subheader(f"{name} ({ticker})")
        
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        summary = info.get('longBusinessSummary', 'Aucune description disponible.')
        
        with st.expander(f"📝 Profil de l'entreprise : {sector} - {industry}"):
            st.write(summary)
            
        # Colonnes principales
        st.markdown("##### 📈 Valorisation & Prix")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capitalisation", format_large_number(info.get('marketCap')), help="Valeur totale de l'entreprise sur le marché (Prix × Nombre d'actions).")
        c2.metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A", help="Price-to-Earnings (PER). Indique combien d'années de bénéfices il faut pour rembourser le prix de l'action. Mesure la cherté de l'action.")
        c3.metric("Price / Sales", f"{info.get('priceToSalesTrailing12Months'):.2f}" if info.get('priceToSalesTrailing12Months') else "N/A", help="Capitalisation divisée par le chiffre d'affaires. Utile pour les entreprises non rentables.")
        c4.metric("EV / EBITDA", f"{info.get('enterpriseToEbitda'):.2f}" if info.get('enterpriseToEbitda') else "N/A", help="Valeur d'Entreprise / EBITDA. Évalue la rentabilité brute indépendamment de la dette ou des impôts.")

        st.markdown("##### 💵 Trésorerie & Dette")
        c5, c6, c7, c8 = st.columns(4)
        de = info.get('debtToEquity')
        c5.metric("Debt / Equity", f"{de:.2f}" if de else "N/A", help="Ratio Dette sur Capitaux Propres. > 100 signifie que l'entreprise a plus de dettes que de fonds propres.")
        c6.metric("Current Ratio", f"{info.get('currentRatio'):.2f}" if info.get('currentRatio') else "N/A", help="Liquidité à court terme (>1 = entreprise saine capable de payer ses factures imminentes).")
        c7.metric("Free Cash Flow", format_large_number(info.get('freeCashflow')), help="Le cash réel qu'il reste à l'entreprise après tout paiement. L'oxygène de l'entreprise.")
        c8.metric("Rendement Dividende", f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/A", help="Le rendement annuel versé en cash directement aux actionnaires.")

        st.markdown("##### 🚀 Rentabilité & Risque")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Marge Nette", f"{info.get('profitMargins')*100:.2f}%" if info.get('profitMargins') else "N/A", help="Sur 100$ de ventes, combien reste-t-il de profit net ? Mesure l'avantage concurrentiel.")
        c10.metric("ROE", f"{info.get('returnOnEquity')*100:.2f}%" if info.get('returnOnEquity') else "N/A", help="Return on Equity : Efficacité de la direction à générer du profit avec l'argent des actionnaires.")
        c11.metric("Beta", f"{info.get('beta'):.2f}" if info.get('beta') else "N/A", help="Volatilité systémique. >1 = Plus risqué et volatil que le marché global.")
        c12.metric("PEG Ratio", f"{info.get('pegRatio'):.2f}" if info.get('pegRatio') else "N/A", help="P/E Ratio divisé par la croissance. < 1 signale une action potentiellement sous-évaluée.")
        
        st.markdown("##### 🕵️ Données Alternatives & Marché")
        c13, c14, c15, c16 = st.columns(4)
        c13.metric("Détention Initiés", f"{info.get('heldPercentInsiders')*100:.2f}%" if info.get('heldPercentInsiders') else "N/A", help="Pourcentage d'actions détenues par les dirigeants et fondateurs. Un chiffre élevé montre qu'ils croient en leur entreprise (Skin in the game).")
        c14.metric("Détention Institutionnelle", f"{info.get('heldPercentInstitutions')*100:.2f}%" if info.get('heldPercentInstitutions') else "N/A", help="Pourcentage d'actions détenues par les Hedge Funds et Banques. Un chiffre élevé = confiance de la 'Smart Money'.")
        c15.metric("52-Week High", f"{info.get('fiftyTwoWeekHigh'):.2f} $" if info.get('fiftyTwoWeekHigh') else "N/A", help="Prix le plus haut atteint sur les 52 dernières semaines.")
        c16.empty()
        
        st.write("---")

def page_strategy_academy():
    render_strategy_academy()

# Fonctions de persistance de portefeuille importées de quant_core.py

def page_paper_trading():
    st.title("🕹️ Simulateur Paper Trading")
    st.markdown("Exécutez vos stratégies en conditions réelles sans risquer votre argent. Solde de départ : **100 000 $**.")
    
    pf = load_portfolio()
    if not isinstance(pf.get('positions'), dict):
        pf['positions'] = {}
    if not isinstance(pf.get('history'), list):
        pf['history'] = []
    
    st.header("💼 Mon Portefeuille Virtuel")
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidités (Cash)", f"{pf['cash']:,.2f} $")
    
    total_positions_value = 0
    if pf['positions']:
        # Fetch current prices for open positions
        tickers_list = list(pf['positions'].keys())
        st.subheader("📈 Positions Ouvertes")
        pos_data = []
        for t in tickers_list:
            try:
                current_price = yf.download(t, period="1d", progress=False)['Close'].iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
            except:
                current_price = pf['positions'][t]['avg_price'] # fallback
            
            qty = pf['positions'][t]['qty']
            avg_price = pf['positions'][t]['avg_price']
            value = qty * current_price
            pnl = value - (qty * avg_price)
            pnl_pct = (pnl / (qty * avg_price)) * 100
            
            total_positions_value += value
            
            with st.expander(f"📦 {get_company_name_from_yahoo(t)} | Valeur: {value:.2f} $ | P&L: {pnl:+.2f} $ ({pnl_pct:+.2f}%)"):
                col1, col2 = st.columns(2)
                col1.write(f"**Quantité:** {qty}")
                col1.write(f"**Prix d'Achat Moyen:** {avg_price:.2f} $")
                col2.write(f"**Prix Actuel:** {current_price:.2f} $")
                
                # --- AVIS EXPERT ---
                st.markdown("---")
                st.markdown("##### 🧠 Avis Expert (Money Management)")
                if pnl_pct >= 20.0:
                    st.success(f"**Take Profit Fortement Recommandé.** Vous avez sécurisé +{pnl_pct:.2f}% de plus-value. En finance quantitative, un tel gain justifie une prise de bénéfices (totale ou partielle) pour lisser la courbe de performance.")
                elif pnl_pct >= 10.0:
                    st.info(f"**Sécurisation Suggérée.** Belle performance de +{pnl_pct:.2f}%. Pensez à remonter votre Stop-Loss (Trailing Stop) au niveau de votre prix d'achat pour garantir un 'Risk-Free Trade' (Trade gratuit).")
                elif pnl_pct <= -15.0:
                    st.error(f"**Stop-Loss Critique !** Perte sévère de {pnl_pct:.2f}%. Les mathématiques du Drawdown sont cruelles : il faudra un gain beaucoup plus fort pour rattraper cette perte. Clôturez la position pour préserver le capital.")
                elif pnl_pct <= -7.0:
                    st.warning(f"**Zone de Danger.** Votre position affiche {pnl_pct:.2f}%. Assurez-vous que votre thèse d'investissement initiale est toujours valide, sinon n'hésitez pas à couper court à la perte.")
                else:
                    st.write(f"**Maintien (Hold).** La position oscille dans sa zone neutre ({pnl_pct:+.2f}%). Laissez le trade respirer pour éviter de sur-réagir au bruit du marché à court terme.")
                st.markdown("---")
                
                if st.button(f"🔴 Liquider {t} au marché", key=f"liq_{t}"):
                    # Simulation Event-Driven (Slippage + Commissions)
                    exec_price = current_price * 0.9990 # -0.10% Slippage
                    gross_revenue = qty * exec_price
                    commission = gross_revenue * 0.0005 # -0.05% Frais
                    net_revenue = gross_revenue - commission
                    
                    pf['cash'] += net_revenue
                    pf['history'].append({
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'action': 'SELL (Manual)',
                        'ticker': t,
                        'qty': qty,
                        'price': exec_price,
                        'total': net_revenue,
                        'pnl': net_revenue - (qty * avg_price)
                    })
                    del pf['positions'][t]
                    save_portfolio(pf)
                    st.success(f"Position {t} liquidée avec succès (Slippage et Frais déduits) !")
                    st.rerun()
    else:
        st.info("Aucune position ouverte actuellement. Utilisez le Terminal de Trading pour acheter.")
        
    c2.metric("Valeur des Positions", f"{total_positions_value:,.2f} $")
    c3.metric("Valeur Totale du Portefeuille", f"{(pf['cash'] + total_positions_value):,.2f} $", delta=f"{((pf['cash'] + total_positions_value) - 100000):+,.2f} $")
    
    st.divider()
    st.header("📜 Historique des Transactions")
    if pf['history']:
        hist_df = pd.DataFrame(pf['history'])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("Aucun historique pour le moment.")
        
    if st.button("⚠️ Réinitialiser le compte (Remise à 100 000 $)"):
        save_portfolio({"cash": 100000.0, "positions": {}, "history": []})
        st.success("Compte réinitialisé.")
        st.rerun()

def page_options_paper_trading():
    st.title("🕹️ Paper Trading (Options & Dérivés)")
    st.markdown("Exécutez vos stratégies d'options (Call/Put) en conditions réelles sans risquer votre argent. Solde de départ indépendant : **100 000 $**.")
    
    pf = load_options_portfolio()
    if not isinstance(pf.get('positions'), dict):
        pf['positions'] = {}
    if not isinstance(pf.get('history'), list):
        pf['history'] = []
    
    st.header("💼 Mon Portefeuille Virtuel (Options)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidités (Cash)", f"{pf['cash']:,.2f} $")
    
    total_positions_value = 0
    if pf['positions']:
        st.subheader("📈 Contrats d'Options Ouverts")
        keys_to_remove = []
        for contract_id, data in pf['positions'].items():
            t = data['ticker']
            qty = data['qty']
            avg_price = data['premium']
            K = data['strike']
            T_days = data['days_to_expiry']
            opt_type = data['type']
            
            # Recalculate price today using Black Scholes
            try:
                df = yf.download(t, period="1y", progress=False)
                current_price = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
                sigma = get_implied_volatility(t, current_price)
            except:
                current_price = data['underlying_price_at_buy']
                sigma = 0.20
                
            T = T_days / 365.0
            if T <= 0.001: T = 0.001 # prevent division by zero
            r = 0.05
            
            # Nouvelle estimation du contrat
            current_premium, delta, gamma, theta, vega = black_scholes(current_price, K, T, r, sigma, opt_type.lower())
            
            # Utilise le multiplicateur enregistré ou tente de le retrouver
            multiplier = data.get('multiplier', get_option_multiplier_and_legislation(t)[0])
            
            value = qty * current_premium * multiplier # *multiplier selon la norme
            buy_value = qty * avg_price * multiplier
            pnl = value - buy_value
            pnl_pct = (pnl / buy_value) * 100 if buy_value > 0 else 0
            
            total_positions_value += value
            
            is_expanded = (st.session_state.get('show_lifecycle_for') == contract_id)
            with st.expander(f"📦 {qty} Contrat(s) {opt_type.upper()} sur {t} | Valeur: {value:.2f} $ | P&L: {pnl:+.2f} $ ({pnl_pct:+.2f}%)", expanded=is_expanded):
                col1, col2 = st.columns(2)
                col1.write(f"**Strike (K):** {K} $")
                col1.write(f"**Prime d'Achat:** {avg_price:.2f} $")
                col1.write(f"**Prix Sous-jacent actuel:** {current_price:.2f} $")
                col1.write(f"**Jours restants:** {T_days}")
                
                col2.write(f"**Prime Actuelle (BS):** {current_premium:.2f} $")
                col2.write(f"**Delta (Direction):** {delta:.3f}")
                col2.write(f"**Gamma (Accélération):** {gamma:.3f}")
                col2.write(f"**Theta (Usure/Temps):** {theta:.3f}")
                col2.write(f"**Vega (Volatilité):** {vega:.3f}")
                
                is_eu = "Européenne" in get_option_multiplier_and_legislation(t)[1]
                can_sell = not (is_eu and T_days > 0)
                
                if not can_sell:
                    st.warning("⚠️ Législation EU : Exercice et Revente bloqués avant l'échéance.")
                
                c_btn1, c_btn2 = st.columns(2)
                
                with c_btn1:
                    if st.button(f"🔴 Revendre le contrat", key=f"liq_opt_{contract_id}", disabled=not can_sell, use_container_width=True):
                        revenue = value
                        pf['cash'] += revenue
                        pf['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'SELL OPTION',
                            'ticker': t,
                            'contract': f"{opt_type.upper()} Strike {K}",
                            'qty': qty,
                            'premium_sold': current_premium,
                            'total_received': revenue,
                            'pnl': pnl
                        })
                        keys_to_remove.append(contract_id)
                        st.success(f"Contrat sur {t} revendu avec succès !")
                
                with c_btn2:
                    if st.button(f"📈 Analyser le Cycle de Vie", key=f"life_btn_{contract_id}", use_container_width=True):
                        if st.session_state.get('show_lifecycle_for') == contract_id:
                            st.session_state['show_lifecycle_for'] = None
                        else:
                            st.session_state['show_lifecycle_for'] = contract_id
                        st.rerun()

                if st.session_state.get('show_lifecycle_for') == contract_id:
                    with st.spinner("⏳ Reconstruction de la timeline du contrat..."):
                        try:
                            # Extraction de la date d'achat du contract_id (format YYYYMMDDHHMMSS)
                            purchase_date_str = contract_id.split('_')[-1]
                            purchase_dt = datetime.strptime(purchase_date_str, "%Y%m%d%H%M%S")
                            start_date = purchase_dt.strftime("%Y-%m-%d")
                            
                            hist_df = yf.download(t, start=start_date, progress=False)
                            
                            if not hist_df.empty:
                                dates = []
                                prices = []
                                bs_premiums = []
                                deltas = []
                                thetas = []
                                
                                for date, row in hist_df.iterrows():
                                    days_passed = (date.tz_localize(None) - purchase_dt).days
                                    if days_passed < 0: days_passed = 0
                                    current_T_days = T_days - days_passed
                                    if current_T_days <= 0: current_T_days = 0.001
                                    
                                    spot = row['Close'].item() if isinstance(row['Close'], pd.Series) else row['Close']
                                    
                                    p, d, g, th, v = black_scholes(spot, K, current_T_days/365.0, r, sigma, opt_type.lower())
                                    
                                    dates.append(date)
                                    prices.append(spot)
                                    bs_premiums.append(p * multiplier * qty)
                                    deltas.append(d)
                                    thetas.append(th)
                                    
                                fig_lc = make_subplots(
                                    rows=4, cols=1, 
                                    shared_xaxes=True, 
                                    vertical_spacing=0.08,
                                    subplot_titles=(
                                        f"Course de l'Action vs Strike ({t})", 
                                        f"Prime du Contrat (Valorisation en $)", 
                                        "Grec Directionnel (Delta)",
                                        "Grec Temporel (Theta $/jour)"
                                    )
                                )
                                
                                # Row 1: Action vs Strike
                                fig_lc.add_trace(go.Scatter(x=dates, y=prices, name="Prix Action", line=dict(color='#00d2ff')), row=1, col=1)
                                fig_lc.add_hline(y=K, line_dash="dash", line_color="red", annotation_text="Strike", row=1, col=1)
                                
                                # Row 2: Prime BS (Totale)
                                fig_lc.add_trace(go.Scatter(x=dates, y=bs_premiums, name="Valeur du Contrat ($)", line=dict(color='#00ff88'), fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'), row=2, col=1)
                                fig_lc.add_hline(y=buy_value, line_dash="dash", line_color="orange", annotation_text="Coût d'Achat", row=2, col=1)
                                
                                # Row 3: Delta
                                fig_lc.add_trace(go.Scatter(x=dates, y=deltas, name="Delta", line=dict(color='#ff00ff')), row=3, col=1)
                                
                                # Row 4: Theta
                                fig_lc.add_trace(go.Scatter(x=dates, y=thetas, name="Theta", line=dict(color='#ff3333'), fill='tozeroy', fillcolor='rgba(255,51,51,0.2)'), row=4, col=1)
                                
                                fig_lc.update_layout(height=900, template="plotly_dark", title_text="🧬 Dashboard Rétroactif du Cycle de Vie", margin=dict(l=20, r=20, t=60, b=20), hovermode="x unified")
                                st.plotly_chart(fig_lc, use_container_width=True)
                            else:
                                st.warning("Pas assez de données historiques pour générer le graphique (Achat récent ou week-end). Revenez demain !")
                        except Exception as e:
                            st.error(f"Impossible de reconstruire le cycle de vie. Soit le contrat est trop ancien (avant cette mise à jour), soit une erreur est survenue : {e}")
        
        for k in keys_to_remove:
            del pf['positions'][k]
        if keys_to_remove:
            save_options_portfolio(pf)
            st.rerun()
            
    else:
        st.info("Aucun contrat ouvert. Utilisez le Pricing d'Options pour acheter.")
        
    c2.metric("Valeur des Contrats", f"{total_positions_value:,.2f} $")
    c3.metric("Valeur Totale du Portefeuille", f"{(pf['cash'] + total_positions_value):,.2f} $", delta=f"{((pf['cash'] + total_positions_value) - 100000):+,.2f} $")
    
    st.divider()
    st.header("📜 Historique des Transactions")
    if pf['history']:
        hist_df = pd.DataFrame(pf['history'])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("Aucun historique pour le moment.")
        
    if st.button("⚠️ Réinitialiser le compte d'Options"):
        save_options_portfolio({"cash": 100000.0, "positions": {}, "history": []})
        st.success("Compte d'Options réinitialisé.")
        st.rerun()

def page_advanced_academy():
    render_advanced_academy()

# black_scholes et find_strike_for_delta sont importés de quant_core.py

def page_options_pricing(tickers):
    st.title("🧮 Pricing d'Options (Black-Scholes)")
    st.markdown("Calculez la valeur théorique (Juste Prix) d'une option d'achat (Call) ou de vente (Put) ainsi que ses paramètres de risque (Les Greeks).")
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans la barre latérale.")
        return
        
    ticker = tickers[0]
    st.header(f"Contrat d'Option sur {ticker}")
    
    # --- IA RECOMMENDATION ---
    st.subheader("🤖 Recommandation Stratégique XGBoost")
    default_opt_index = 0
    if f"trader_{ticker}" in st.session_state:
        trader = st.session_state[f"trader_{ticker}"]
        if trader.is_trained:
            # Re-predict using latest data
            df_raw = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.droplevel(1)
            macro_df = get_macro_data("1y", "1d")
            df = add_features(df_raw)
            if macro_df is not None:
                df = join_macro_data(df, macro_df)
            
            last_row = df.iloc[-1:]
            features = trader.get_feature_list(df)
            prob = trader.predict(last_row, features)
            
            if prob > 0.6:
                st.success(f"📈 L'IA est fortement **haussière** sur {ticker} (Confiance {prob:.0%}). **Stratégie recommandée : Achat de CALL.**")
                default_opt_index = 0
            elif prob < 0.4:
                st.error(f"📉 L'IA est fortement **baissière** sur {ticker} (Confiance {(1-prob):.0%}). **Stratégie recommandée : Achat de PUT.**")
                default_opt_index = 1
            else:
                st.warning(f"⚖️ L'IA est **neutre** sur {ticker}. Marché incertain, évitez les options directionnelles.")
                default_opt_index = 0
        else:
            st.info("Le modèle IA est présent mais non entraîné.")
            default_opt_index = 0
    else:
        st.info("💡 Entraînez d'abord l'IA dans le Terminal de Trading pour obtenir une recommandation quant à la direction de l'option.")
        default_opt_index = 0

    st.divider()
    
    try:
        df = yf.download(ticker, period="1y", progress=False)
        S = df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        sigma_hist = returns.std() * np.sqrt(252) # Volatilité annualisée
        if isinstance(sigma_hist, pd.Series):
            sigma_hist = sigma_hist.iloc[0].item()
        elif hasattr(sigma_hist, 'item'):
            sigma_hist = sigma_hist.item()
    except:
        S = 100.0
        sigma_hist = 0.20
        
    col1, col2 = st.columns(2)
    with col1:
        S_input = st.number_input("Prix de l'action (Spot - S)", value=float(S))
        K = st.number_input("Prix d'exercice (Strike - K)", value=float(S))
        T_days = st.slider("Jours avant expiration", 1, 365, 30)
        T = T_days / 365.0
    with col2:
        sigma = st.slider("Volatilité Implicite (σ)", 0.01, 1.0, float(sigma_hist))
        r = st.slider("Taux d'intérêt sans risque (r)", 0.0, 0.10, 0.05)
        opt_type = st.radio("Type d'Option", ["Call", "Put"], index=default_opt_index)
        
    price, delta, gamma, theta, vega = black_scholes(S_input, K, T, r, sigma, opt_type.lower())
    multiplier, legislation = get_option_multiplier_and_legislation(ticker)
    
    st.divider()
    st.subheader(f"Valeur Théorique de l'Option (Black-Scholes) : **{price:.2f} $** par action")
    
    st.info(f"💡 **Prime vs Contrat :** La valeur ci-dessus est la 'Prime' calculée pour **1 action**. Or, la norme de ce titre ({legislation}) impose des lots de **{multiplier} actions**.\n\n👉 **Le coût réel d'un contrat (1 lot) sera donc de : {price:.2f} $ × {multiplier} = {price * multiplier:.2f} $**")
    st.write("### Les Greeks (Paramètres de Risque)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{delta:.3f}", help="Sensibilité au prix. Si l'action monte de 1$, l'option prendra Delta $. (0 à 1 pour Call, -1 à 0 pour Put).")
    g2.metric("Gamma (Γ)", f"{gamma:.4f}", help="Vitesse du Delta. De combien le Delta change si l'action monte de 1$.")
    g3.metric("Theta (Θ)", f"{theta:.3f} $/jour", help="Érosion du temps. Combien l'option perd de valeur chaque jour qui passe (Time Decay).")
    g4.metric("Vega (ν)", f"{vega/100:.3f}", help="Sensibilité à la volatilité. Si la volatilité augmente de 1%, l'option prend Vega $.")

    # --- GRAPHIQUE DE PAYOFF ---
    st.divider()
    st.subheader("📊 Graphique de Payoff à l'expiration")
    st.markdown(f"**Pourquoi le graphique commence-t-il dans le rouge ?** Parce que pour obtenir ce droit d'acheter/vendre, vous avez payé la Prime aujourd'hui ({price:.2f} $ par action). Pour être rentable au moment de l'expiration, l'action devra donc bouger suffisamment pour rembourser cette prime : c'est le **Break-Even (Seuil de Rentabilité)**.")
    
    # Générer des prix possibles à l'expiration (±30% autour du strike)
    prices_range = np.linspace(K * 0.7, K * 1.3, 100)
    
    if opt_type.lower() == "call":
        payoff = np.maximum(prices_range - K, 0) - price
        breakeven = K + price
    else:
        payoff = np.maximum(K - prices_range, 0) - price
        breakeven = K - price
        
    fig_payoff = go.Figure()
    # Zone de profit (vert) et zone de perte (rouge)
    fig_payoff.add_trace(go.Scatter(
        x=prices_range, y=payoff,
        mode='lines',
        name='Profit/Perte',
        line=dict(color='white', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)' if opt_type.lower() == "call" else 'rgba(255, 0, 0, 0.2)'
    ))
    
    # Ligne 0
    fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Point de Break-even
    fig_payoff.add_vline(x=breakeven, line_dash="dash", line_color="yellow", annotation_text=f"Break-even: {breakeven:.2f} $")
    
    # Prix actuel (Spot)
    fig_payoff.add_vline(x=S_input, line_dash="dot", line_color="blue", annotation_text=f"Spot Actuel: {S_input:.2f} $")

    # Mettre en rouge la partie négative
    payoff_negative = np.where(payoff < 0, payoff, 0)
    fig_payoff.add_trace(go.Scatter(
        x=prices_range, y=payoff_negative,
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.5)',
        showlegend=False
    ))
    
    fig_payoff.update_layout(
        title=f"Profil de Profit & Perte (Achat de {opt_type})",
        xaxis_title="Prix du sous-jacent à l'expiration ($)",
        yaxis_title="Profit / Perte ($)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

    # --- ACHAT VIRTUEL ---
    st.divider()
    st.subheader("🛒 Exécution Virtuelle (Paper Trading)")
    st.write(f"**Norme applicable :** {legislation}")
    qty_options = st.number_input(f"Combien de contrats souhaitez-vous acheter ? (1 contrat = {multiplier} actions)", min_value=1, value=1)
    cout_total = qty_options * price * multiplier
    st.info(f"💰 **Total débité de votre compte virtuel :** {qty_options} contrats × {price*multiplier:.2f} $ = **{cout_total:.2f} $**")
    
    if st.button("✅ Acheter ce contrat (Paper Trading)", use_container_width=True):
        pf_opt = load_options_portfolio()
        if not isinstance(pf_opt.get('positions'), dict):
            pf_opt['positions'] = {}
        if not isinstance(pf_opt.get('history'), list):
            pf_opt['history'] = []
        if pf_opt.get('cash', 0.0) >= cout_total:
            pf_opt['cash'] -= cout_total
            contract_id = f"{ticker}_{opt_type.upper()}_{K}_{T_days}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            pf_opt['positions'][contract_id] = {
                'ticker': ticker,
                'type': opt_type.lower(),
                'strike': K,
                'days_to_expiry': T_days,
                'premium': price,
                'qty': qty_options,
                'multiplier': multiplier,
                'underlying_price_at_buy': S_input,
                'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pf_opt['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY OPTION',
                'ticker': ticker,
                'contract': f"{opt_type.upper()} Strike {K}",
                'qty': qty_options,
                'premium_paid': price,
                'total_cost': cout_total
            })
            save_options_portfolio(pf_opt)
            st.success(f"Contrat acheté avec succès ! Retrouvez-le dans l'onglet 'Paper Trading (Options)'.")
            st.balloons()
        else:
            st.error("Fonds insuffisants dans votre portefeuille d'options virtuel.")

    # --- SURFACE DE VOLATILITÉ ET ANOMALIES ---
    st.divider()
    st.header("🌊 Surface de Volatilité & Scanner d'Anomalies (Live Market)")
    st.markdown("Récupération en temps réel des contrats d'options pour tracer le **Volatility Smile** 3D et détecter les options **Surévaluées/Sous-évaluées** par le marché.")
    
    if st.button("🔍 Lancer le Scanner d'Options en Direct", use_container_width=True):
        with st.spinner("Téléchargement de la chaîne d'options depuis le marché (Cela peut prendre quelques secondes)..."):
            try:
                ticker_obj = yf.Ticker(ticker)
                expirations = ticker_obj.options
                
                if not expirations:
                    st.warning("Aucune chaîne d'options disponible pour cette action sur le marché public.")
                else:
                    data_calls = []
                    # On limite aux 5 prochaines échéances pour ne pas figer l'application
                    for exp in expirations[:5]:
                        try:
                            chain = ticker_obj.option_chain(exp)
                            calls = chain.calls
                            
                            days_to_exp = (pd.to_datetime(exp) - datetime.now()).days
                            if days_to_exp <= 0:
                                continue
                                
                            for _, row in calls.iterrows():
                                strike = row['strike']
                                # Filtre pour ne garder que les strikes proches de la monnaie (± 30%)
                                if 0.7 * S_input <= strike <= 1.3 * S_input:
                                    iv = row['impliedVolatility']
                                    bid = row['bid']
                                    ask = row['ask']
                                    mid_price = (bid + ask) / 2.0
                                    
                                    # Calcul du prix théorique avec la Volatilité Historique (sigma_hist) pour trouver l'anomalie
                                    theo_price = black_scholes(S_input, strike, days_to_exp/365.0, r, sigma_hist, 'call')[0]
                                    
                                    if theo_price > 0:
                                        overval_pct = ((mid_price - theo_price) / theo_price) * 100
                                    else:
                                        overval_pct = 0
                                        
                                    data_calls.append({
                                        'Strike': strike,
                                        'Jours_Expiration': days_to_exp,
                                        'IV': iv,
                                        'Prix_Marché': mid_price,
                                        'Prix_Théorique': theo_price,
                                        'Écart_Surévaluation_%': overval_pct
                                    })
                        except:
                            continue
                    
                    if data_calls:
                        df_surf = pd.DataFrame(data_calls)
                        
                        st.subheader("📊 Surface de Volatilité 3D (Implied Volatility)")
                        fig_3d = go.Figure(data=[go.Mesh3d(
                            x=df_surf['Strike'],
                            y=df_surf['Jours_Expiration'],
                            z=df_surf['IV'],
                            intensity=df_surf['IV'],
                            colorscale='Viridis',
                            opacity=0.8
                        )])
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title="Strike ($)",
                                yaxis_title="Jours avant Expiration",
                                zaxis_title="Volatilité Implicite (σ)"
                            ),
                            template="plotly_dark",
                            height=600,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                        
                        st.subheader("🚨 Scanner d'Anomalies (Opportunités d'Arbitrage)")
                        st.markdown("Ce tableau compare le Prix du Marché (Mid-Price) au Prix Théorique de Black-Scholes. Les contrats fortement surévalués sont d'excellents candidats pour des stratégies de vente (Short Call).")
                        
                        # Trier par les contrats les plus surévalués
                        df_anomalies = df_surf.sort_values(by='Écart_Surévaluation_%', ascending=False).head(15)
                        
                        # Formatage visuel avec Pandas Styler
                        st.dataframe(
                            df_anomalies.style.format({
                                'Strike': '{:.2f} $',
                                'IV': '{:.2%}',
                                'Prix_Marché': '{:.2f} $',
                                'Prix_Théorique': '{:.2f} $',
                                'Écart_Surévaluation_%': '{:+.2f} %'
                            }).background_gradient(subset=['Écart_Surévaluation_%'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                    else:
                        st.error("Erreur lors de l'extraction des données ou aucun contrat liquide trouvé.")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API de marché : {e}")

def page_options_academy():
    render_options_academy()

def page_rag_gemini(tickers, gemini_api_key):
    st.title("🧠 Analyse Fondamentale & RAG (Gemini AI)")
    st.markdown("""
    Ce module utilise l'Intelligence Artificielle Générative (Google Gemini) pour agir comme un **Analyste Quantitatif**. 
    L'IA va lire les actualités récentes, les données financières, et rédiger un rapport d'expertise pour anticiper des événements comme l'**IV Crush**.
    """)
    
    if not genai:
        st.error("❌ La librairie `google-genai` n'est pas installée. Le système fonctionnera si vous l'installez via le terminal (`pip install google-genai`).")
        return
        
    if not gemini_api_key:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        
    if not gemini_api_key:
        st.warning("⚠️ Clé API Gemini non configurée.")
        st.info("""
        ### 🔑 Comment obtenir votre Clé API Gemini 100% Gratuite en 30 secondes :
        1. Rendez-vous sur le portail officiel : **[Google AI Studio (Lien direct)](https://aistudio.google.com/app/apikey)**.
        2. Connectez-vous avec votre compte Google (aucun moyen de paiement ni carte bancaire requis).
        3. Cliquez sur le bouton bleu **"Create API key"** -> **"Create API key in new project"**.
        4. Copiez votre clé (commençant par `AIzaSy...`) et collez-la dans le champ `🔑 Clé API Gemini` dans le menu latéral à gauche.
        
        💡 **Astuce Serveur / VPS** : Vous pouvez également définir la variable d'environnement `GEMINI_API_KEY=AIzaSy...` sur votre machine ou dans un fichier `.env`. L'application la détectera alors automatiquement sans aucune saisie manuelle !
        """)
        return
    # Configure inside the loop now
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral.")
        return
        
    selected_ticker = st.selectbox("Sélectionnez l'action à analyser", tickers)
    
    if st.button(f"🔍 Lancer l'Analyse Stratégique sur {selected_ticker}", use_container_width=True):
        with st.spinner(f"Récupération des données et réflexion de l'IA sur {selected_ticker}..."):
            try:
                stock = yf.Ticker(selected_ticker)
                
                # 1. Retrieval (Extraction)
                info = stock.info
                news = stock.news
                
                company_name = info.get('longName', selected_ticker)
                sector = info.get('sector', 'Inconnu')
                industry = info.get('industry', 'Inconnu')
                forward_pe = info.get('forwardPE', 'N/A')
                div_yield = info.get('dividendYield', 'N/A')
                
                # Format news
                news_text = ""
                if news:
                    for n in news[:5]:
                        # Handle different formats returned by yfinance news
                        title = n.get('content', {}).get('title', n.get('title', ''))
                        summary = n.get('content', {}).get('summary', n.get('summary', ''))
                        if not summary:
                            summary = n.get('publisher', '')
                        news_text += f"- Titre : {title}\n  Contexte : {summary}\n\n"
                else:
                    news_text = "Aucune actualité récente trouvée."
                    
                # 1.5 Calcul Quantitatif Réel (XGBoost)
                prob_text = "Non calculé"
                try:
                    df_raw_quant = yf.download(selected_ticker, period="2y", interval="1d", progress=False)
                    if not df_raw_quant.empty:
                        if isinstance(df_raw_quant.columns, pd.MultiIndex):
                            df_raw_quant.columns = df_raw_quant.columns.droplevel(1)
                        df_raw_quant = df_raw_quant.loc[:, ~df_raw_quant.columns.duplicated()].copy()
                        df_quant = add_features(df_raw_quant)
                        macro_df_q = get_macro_data(period="2y", interval="1d")
                        if macro_df_q is not None:
                            df_quant = df_quant.join(macro_df_q, how='left').ffill()
                        df_quant = df_quant.dropna()
                        
                        trader_key = f"port_trader_{selected_ticker}"
                        if trader_key in st.session_state and st.session_state[trader_key].is_trained:
                            trader_quant = st.session_state[trader_key]
                        else:
                            trader_quant = MLTrader()
                            features_quant = trader_quant.train(df_quant, optimize=False, use_wfa=False)
                            
                        features_to_use = trader_quant.get_feature_list(df_quant)
                        prob_quant = trader_quant.predict(df_quant.iloc[-1:], features_to_use)
                        if prob_quant is not None:
                            direction = "HAUSSIÈRE" if prob_quant > 0.55 else ("BAISSIÈRE" if prob_quant < 0.45 else "NEUTRE")
                            prob_text = f"{prob_quant:.1%} de probabilité {direction}"
                except Exception as ex_q:
                    prob_text = f"Calcul indisponible ({ex_q})"

                # 2. Le Prompt
                prompt = f"""
Agis comme un Trader Quantitatif Institutionnel expert en Options et Dérivés.
Ton objectif est de rédiger un rapport stratégique d'analyse fondamentale sur l'action {company_name} (Ticker: {selected_ticker}).

Voici les informations extraites aujourd'hui (Retrieval) :
- Secteur : {sector} ({industry})
- Forward P/E : {forward_pe}
- Dividend Yield : {div_yield}

Dernières actualités de la société (très important) :
{news_text}

Information Cruciale - Prédiction Quantitative (XGBoost) :
Notre algorithme de Machine Learning (XGBoost) vient d'analyser techniquement cette action.
Prédiction mathématique : {prob_text}

Ta mission est de fusionner ces deux visions pour devenir le Portfolio Manager ultime. Rédige un rapport structuré en français (Markdown) contenant :
1. **Le Sentiment Profond :** Quelle est l'ambiance générale autour de l'entreprise au vu des news ? Y a-t-il un momentum caché ?
2. **Confrontation IA vs Marché :** La prédiction mathématique de XGBoost est-elle alignée avec les fondamentaux actuels, ou y a-t-il une divergence dangereuse ?
3. **Risque d'IV Crush & Volatilité :** Au vu des actus, y a-t-il un événement imminent qui pourrait causer un pic d'Implied Volatility (IV) suivi d'un écrasement brutal ?
4. **Stratégie d'Options Recommandée :** En fonction du sentiment, de XGBoost et de la volatilité, quelle stratégie d'options recommanderais-tu (ex: Vendre des Puts, Iron Condor, etc.) et pourquoi ?

Ne mets pas de blabla d'introduction de chatbot, va droit au but avec un ton très professionnel, technique et institutionnel. Utilise du gras et des listes à puces.
"""
                # 3. Generation
                success = False
                all_errors = {}
                # Modèles récents et stables recommandés par l'API Google en 2026
                models_to_try = [
                    'gemini-3.8-flash',
                    'gemini-3.1-pro-preview',
                    'gemini-3.1-pro',
                    'gemini-2.0-flash',
                    'gemini-2.5-flash',
                    'gemini-2.5-pro',
                    'gemini-1.5-flash'
                ]
                
                response = None
                client = genai.Client(api_key=gemini_api_key.strip())
                for m_name in models_to_try:
                    try:
                        response = client.models.generate_content(model=m_name, contents=prompt)
                        st.info(f"Modèle utilisé avec succès : `{m_name}`")
                        success = True
                        break
                    except Exception as e:
                        all_errors[m_name] = str(e)
                        continue
                        
                if not success:
                    st.error("Aucun modèle Gemini n'a pu traiter la requête.")
                    
                    # Détection spécifique du blocage de service ou clé restreinte
                    if any("API_KEY_SERVICE_BLOCKED" in str(err) for err in all_errors.values()):
                        st.error("""
                        ### 🔑 Diagnostic : Votre clé API a le service Gemini bloqué (Code 403 - API_KEY_SERVICE_BLOCKED)
                        
                        Ce blocage est causé par la configuration de votre clé sur Google Cloud (Projet `283448164473`) :
                        
                        1. **Méthode Recommandée (La plus rapide - 30 sec)** :
                           - Rendez-vous sur **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
                           - Cliquez sur **"Create API Key"** -> **"Create API key in new project"**.
                           - Copiez cette clé neuve (elle n'a aucune restriction et l'API Gemini est activée par défaut).
                           - Collez-la dans la barre latérale sous `🔑 Clé API Gemini`.
                        
                        2. **Si vous tenez à conserver votre projet Google Cloud existant** :
                           - **Activez l'API** : Allez sur [Activer Generative Language API](https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com) et cliquez sur **ACTIVER**.
                           - **Supprimez les restrictions de la clé** : Dans Google Cloud Console > Identifiants > Cliquez sur votre clé API. Sous *Restrictions d'API*, cochez **"Ne pas restreindre la clé"** (ou ajoutez `Generative Language API` à la liste autorisée).
                        """)
                    else:
                        for m, err in all_errors.items():
                            st.warning(f"**{m}** : {err}")
                    return
                st.success("Analyse terminée avec succès !")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {str(e)}")

def page_bot_config():
    st.title("🤖 Paramétrage du Bot (Headless)")
    st.markdown("Configurez ici l'automatisation. Le bot s'exécutera en tâche de fond tous les soirs, s'entraînera, et vous enverra un signal via Telegram ou Discord.")
    
    BOT_CONFIG_FILE = "bot_config.json"
    
    # Charger la conf existante
    if os.path.exists(BOT_CONFIG_FILE):
        with open(BOT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "webhook_discord": "",
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "tickers": ["Apple Inc. (US)"],
            "run_time": "22:00",
            "is_active": False
        }
        
    st.header("1. Cibles d'Analyse Nocturne")
    
    portfolio_choice = st.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0
    )
    
    if portfolio_choice != "Sélection Manuelle":
        default_selection = PREDEFINED_PORTFOLIOS[portfolio_choice]
    else:
        default_selection = config.get("tickers", ["Apple Inc. (US)"])
        
    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    
    stock_choices = st.multiselect(
        "Rechercher une ou plusieurs actions",
        options=list(MAJOR_STOCKS.keys())[1:],
        default=valid_defaults
    )
    
    selected_tickers = []
    selected_tickers.extend(stock_choices)
    
    # Ajout silencieux des tickers personnalisés du portefeuille (ex: Chine)
    for s in default_selection:
        if s not in MAJOR_STOCKS.keys() and s not in selected_tickers:
            selected_tickers.append(s)
    
    st.header("2. Canaux d'Alertes")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Discord Webhook")
        webhook_discord = st.text_input("URL du Webhook Discord", value=config.get("webhook_discord", ""))
    with c2:
        st.subheader("Telegram Bot API")
        telegram_token = st.text_input("Bot Token", value=config.get("telegram_bot_token", ""))
        telegram_chat = st.text_input("Chat ID", value=config.get("telegram_chat_id", ""))
        
    st.header("3. Planification")
    run_time_val = config.get("run_time", "22:00")
    run_time_obj = datetime.strptime(run_time_val, "%H:%M").time()
    run_time = st.time_input("Heure d'exécution quotidienne", value=run_time_obj)
    
    if st.button("💾 Sauvegarder la Configuration"):
        config["webhook_discord"] = webhook_discord
        config["telegram_bot_token"] = telegram_token
        config["telegram_chat_id"] = telegram_chat
        config["tickers"] = selected_tickers
        config["run_time"] = run_time.strftime("%H:%M")
        with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        st.success("Configuration sauvegardée !")
        
    st.divider()
    st.header("4. Contrôle du Serveur (Daemon)")
    
    if config.get("is_active", False):
        st.success("🟢 Le Bot Headless est actuellement **ACTIF**.")
        if st.button("🛑 Arrêter le Bot"):
            config["is_active"] = False
            with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            st.rerun()
    else:
        st.warning("🔴 Le Bot Headless est actuellement **ARRÊTÉ**.")
        if st.button("🚀 Démarrer le Bot en Arrière-plan"):
            config["is_active"] = True
            with open(BOT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            
            # Lancer le sous-processus sans bloquer
            if os.name == 'nt': # Windows
                subprocess.Popen(["python", "headless_bot.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else: # Linux/Mac
                subprocess.Popen(["python3", "headless_bot.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            st.success("Bot démarré en tâche de fond ! Il s'entraînera chaque soir.")
            st.rerun()

def page_options_backtester(tickers):
    st.title("⏪ Backtester d'Options (Simulation Quantitative)")
    st.markdown("""
    Ce module est le **Graal** des Quants. Il permet de simuler des **Stratégies de Rente Systématiques** (Yield Generation).
    Le moteur mathématique reconstruit l'historique des prix des options passées en ciblant un Delta précis, et simule l'ouverture et la fermeture des contrats jour par jour.
    """)
    
    if not tickers:
        st.warning("Veuillez sélectionner au moins une action dans le menu latéral.")
        return
        
    ticker = tickers[0]
    st.header(f"Configuration du Backtest sur {ticker}")
    
    c1, c2, c3, c4 = st.columns(4)
    strategy = c1.selectbox("Stratégie", ["Single Leg", "Credit Spread", "Straddle", "Iron Condor"])
    strat_type = c2.selectbox("Direction (Type)", ["PUT", "CALL"]) if strategy in ["Single Leg", "Credit Spread"] else None
    strat_action = c3.selectbox("Action", ["Vendre (Short)", "Acheter (Long)"]) if strategy == "Single Leg" else None
    target_delta_abs = c4.number_input("Delta Cible (absolu)", min_value=0.01, max_value=0.99, value=0.15, step=0.01, help="Delta ciblé pour les options vendues (ex: 0.15)")
    target_dte = st.number_input("DTE (Jours à l'échéance)", min_value=1, max_value=365, value=45, step=1)
    
    c5, c6, c7, c8 = st.columns(4)
    take_profit_pct = c5.number_input("Take Profit (% de la prime max)", min_value=10, max_value=100, value=50, step=5)
    stop_loss_pct = c6.number_input("Stop Loss (% de la prime max)", min_value=10, max_value=1000, value=200, step=10)
    capital = c7.number_input("Capital Initial ($)", value=100000, step=10000)
    period = c8.selectbox("Période d'historique", ["1y", "2y", "5y", "10y"], index=2)
    
    if st.button("🚀 Lancer le Backtest Systématique", use_container_width=True):
        with st.spinner("Téléchargement de l'historique et simulation des options..."):
            try:
                df = yf.download(ticker, period=period, progress=False)
                if df.empty:
                    st.error("Pas de données trouvées.")
                    return
                
                closes = df['Close']
                if isinstance(closes, pd.DataFrame): closes = closes.iloc[:, 0]
                
                # Volatilité historique glissante (30 jours)
                returns = closes.pct_change()
                hist_vol = returns.rolling(window=30).std() * np.sqrt(252)
                hist_vol = hist_vol.bfill()
                
                r = 0.05
                multiplier, _ = get_option_multiplier_and_legislation(ticker)
                
                cash = capital
                equity_curve = []
                trades = []
                current_position = None
                
                pb = st.progress(0)
                total_days = len(closes)
                
                for i, (date, spot) in enumerate(closes.items()):
                    if i % 50 == 0: pb.progress(i / total_days)
                    sigma = hist_vol.iloc[i]
                    if sigma < 0.05: sigma = 0.05
                    
                    if current_position is None:
                        # OPEN POSITION
                        legs = []
                        if strategy == "Single Leg":
                            td = target_delta_abs if strat_type == "CALL" else -target_delta_abs
                            K = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td)
                            legs.append({'type': strat_type.lower(), 'action': 'short' if 'Vendre' in strat_action else 'long', 'strike': K})
                        elif strategy == "Credit Spread":
                            td_short = target_delta_abs if strat_type == "CALL" else -target_delta_abs
                            td_long = (target_delta_abs/2.0) if strat_type == "CALL" else -(target_delta_abs/2.0)
                            K_short = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td_short)
                            K_long = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td_long)
                            legs.append({'type': strat_type.lower(), 'action': 'short', 'strike': K_short})
                            legs.append({'type': strat_type.lower(), 'action': 'long', 'strike': K_long})
                        elif strategy == "Straddle":
                            legs.append({'type': 'call', 'action': 'long', 'strike': spot})
                            legs.append({'type': 'put', 'action': 'long', 'strike': spot})
                        elif strategy == "Iron Condor":
                            Kc_s = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'call', target_delta_abs)
                            Kc_l = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'call', target_delta_abs/2.0)
                            Kp_s = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'put', -target_delta_abs)
                            Kp_l = find_strike_for_delta(spot, target_dte/365.0, r, sigma, 'put', -target_delta_abs/2.0)
                            
                            legs.append({'type': 'call', 'action': 'short', 'strike': Kc_s})
                            legs.append({'type': 'call', 'action': 'long', 'strike': Kc_l})
                            legs.append({'type': 'put', 'action': 'short', 'strike': Kp_s})
                            legs.append({'type': 'put', 'action': 'long', 'strike': Kp_l})
                            
                        total_cost_in = 0
                        for leg in legs:
                            p, _, _, _, _ = black_scholes(spot, leg['strike'], target_dte/365.0, r, sigma, leg['type'])
                            leg['premium_in'] = p
                            leg_cost = p * multiplier
                            if leg['action'] == 'short':
                                total_cost_in += leg_cost
                            else:
                                total_cost_in -= leg_cost
                                
                        current_position = {
                            'entry_date': date,
                            'entry_spot': spot,
                            'legs': legs,
                            'net_cost_in': total_cost_in,
                            'days_passed': 0,
                        }
                        
                        cash += total_cost_in
                        equity_curve.append(cash)
                    
                    else:
                        # MANAGE POSITION
                        current_position['days_passed'] += 1
                        days_remaining = target_dte - current_position['days_passed']
                        
                        if days_remaining <= 0:
                            # EXPIRATION
                            closing_cashflow = 0
                            for leg in current_position['legs']:
                                intrinsic = max(0, spot - leg['strike']) if leg['type'] == 'call' else max(0, leg['strike'] - spot)
                                val = intrinsic * multiplier
                                closing_cashflow += (-val if leg['action'] == 'short' else val)
                                    
                            cash += closing_cashflow
                            pnl = current_position['net_cost_in'] + closing_cashflow
                            
                            trades.append({
                                'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                                'exit_date': date.strftime('%Y-%m-%d'),
                                'strategy': strategy,
                                'reason': 'Expiration',
                                'pnl': pnl
                            })
                            current_position = None
                            equity_curve.append(cash)
                            
                        else:
                            # CHECK TP / SL
                            closing_cashflow = 0
                            for leg in current_position['legs']:
                                p, _, _, _, _ = black_scholes(spot, leg['strike'], days_remaining/365.0, r, sigma, leg['type'])
                                val = p * multiplier
                                closing_cashflow += (-val if leg['action'] == 'short' else val)
                                    
                            current_pnl = current_position['net_cost_in'] + closing_cashflow
                            
                            close_position = False
                            reason = ""
                            
                            if current_position['net_cost_in'] > 0: # Stratégie Crédit (Vente)
                                max_profit = current_position['net_cost_in']
                                if current_pnl >= max_profit * (take_profit_pct / 100.0):
                                    close_position = True
                                    reason = "Take Profit"
                                elif current_pnl <= -max_profit * (stop_loss_pct / 100.0):
                                    close_position = True
                                    reason = "Stop Loss"
                            else: # Stratégie Débit (Achat)
                                max_loss = abs(current_position['net_cost_in'])
                                if max_loss > 0:
                                    if current_pnl >= max_loss * (take_profit_pct / 100.0):
                                        close_position = True
                                        reason = "Take Profit"
                                    elif current_pnl <= -max_loss * (stop_loss_pct / 100.0):
                                        close_position = True
                                        reason = "Stop Loss"
                                        
                            if close_position:
                                cash += closing_cashflow
                                trades.append({
                                    'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                                    'exit_date': date.strftime('%Y-%m-%d'),
                                    'strategy': strategy,
                                    'reason': reason,
                                    'pnl': current_pnl
                                })
                                current_position = None
                                equity_curve.append(cash)
                            else:
                                mtm_equity = cash + closing_cashflow
                                equity_curve.append(mtm_equity)
                
                pb.empty()
                
                # --- RESULTS ---
                st.divider()
                st.header("📊 Résultats de la Simulation")
                
                if not trades:
                    st.warning("Aucun trade n'a pu être clôturé sur cette période.")
                    return
                    
                trades_df = pd.DataFrame(trades)
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['pnl'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                total_pnl = trades_df['pnl'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Capital Final", f"{equity_curve[-1]:,.2f} $", f"{(equity_curve[-1] - capital):+,.2f} $")
                col2.metric("Total Trades", total_trades)
                col3.metric("Win Rate", f"{win_rate:.2%}")
                col4.metric("Profit Net Moyen / Trade", f"{(total_pnl / total_trades):.2f} $")
                
                dates = closes.index
                bnh = (closes / closes.iloc[0]) * capital
                
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Scatter(x=dates, y=equity_curve, name="Stratégie Options", line=dict(color='#00ff88', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=bnh, name="Buy & Hold (Action)", line=dict(color='#00d2ff', width=1, dash='dot')))
                
                fig.update_layout(title="Évolution du Portefeuille (Equity Curve)", template="plotly_dark", height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📝 Journal des Transactions (Trades Log)"):
                    st.dataframe(trades_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Erreur durant la simulation : {e}")

# --- FONCTION PRINCIPALE ---
def main():
    st.sidebar.title("🧭 Menu Principal")
    menu = st.sidebar.radio("Sélectionnez un module :", [
        "📈 Terminal de Trading",
        "🕹️ Paper Trading (Virtuel)",
        "🕹️ Paper Trading (Options)",
        "⏪ Backtester Systématique (Options)",
        "🤖 Paramétrage du Bot (Headless)",
        "🧠 Analyse RAG (Gemini)",
        "🏢 Fondamentaux Financiers",
        "🧮 Options & Dérivés (Pricing)",
        "🎓 Académie: Stratégie & Risques",
        "🎓 Académie: Modélisation Avancée",
        "🎓 Académie: Options & Dérivés",
        "📚 Académie: Indicateurs",
        "🤖 Académie: XGBoost",
        "📰 Académie: News & Sentiment"
    ])
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ Configuration Globale")
    
    default_gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_api_key = st.sidebar.text_input(
        "🔑 Clé API Gemini (Optionnel)", 
        value=default_gemini_key, 
        type="password", 
        help="Nécessaire uniquement pour le module d'Analyse RAG. Obtenez une clé 100% gratuite sur https://aistudio.google.com/app/apikey"
    )
    
    
    # --- NOUVEAU : Sélection rapide de portefeuilles ---
    portfolio_choice = st.sidebar.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0
    )
    
    # Si un portefeuille est choisi, on utilise sa liste. Sinon par défaut Apple.
    default_selection = PREDEFINED_PORTFOLIOS[portfolio_choice] if portfolio_choice != "Sélection Manuelle" else ["Apple Inc. (US)"]
    
    # Filtrer les valeurs par défaut qui sont dans MAJOR_STOCKS pour le multiselect
    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    
    # Remplacement par Multiselect réactif
    stock_choices = st.sidebar.multiselect(
        "Rechercher une ou plusieurs actions", 
        options=list(MAJOR_STOCKS.keys()), 
        default=valid_defaults,
        key=f"stocks_ms_{portfolio_choice}"
    )
    
    tickers = []
    
    # Ajout silencieux des tickers qui ne sont pas dans MAJOR_STOCKS (ex: Chine)
    for s in default_selection:
        if s not in MAJOR_STOCKS.keys():
            tickers.append(s)

    if "--- Saisir manuellement ---" in stock_choices:
        custom_ticker = st.sidebar.text_input("Ticker personnalisé (ex: EPA:MC)", "AAPL")
        tickers.append(convert_google_to_yahoo_ticker(custom_ticker))
        
    for choice in stock_choices:
        if choice != "--- Saisir manuellement ---":
            tickers.append(convert_google_to_yahoo_ticker(MAJOR_STOCKS[choice]))
            
    # Déduplication
    tickers = list(set(tickers))
    
    if menu == "📈 Terminal de Trading":
        st.title("🚀 XGBoost Stock Trader Pro")
        st.sidebar.header("Paramètres IA")
        
        period = st.sidebar.selectbox("Période d'historique", ["2y", "5y", "10y", "max"], index=3)
        interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)
        initial_capital = st.sidebar.number_input("Capital Initial Total ($)", min_value=100, max_value=1000000, value=10000, step=100)
        optimize_model = st.sidebar.checkbox("🧠 Auto-Optimisation (Optuna Bayésien)")
        
        st.sidebar.divider()
        st.sidebar.header("Logique Walk-Forward (WFA)")
        use_wfa = st.sidebar.checkbox("Activer Walk-Forward Analysis (WFA)")
        wfa_train_window = st.sidebar.selectbox("Fenêtre Entraînement", ["2Y", "5Y", "7Y"], index=1)
        wfa_step = st.sidebar.selectbox("Pas Avancement", ["1M", "2M", "6M", "1Y"], index=2)
        
        # WFA Fin = Date d'aujourd'hui, WFA Début = WFA Fin - contenu de la fenêtre Entraînement (2Y, 5Y, 7Y)
        window_offsets = {"2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5), "7Y": pd.DateOffset(years=7)}
        wfa_offset = window_offsets.get(wfa_train_window, pd.DateOffset(years=5))
        
        default_wfa_end = datetime.now().date()
        default_wfa_start = (pd.Timestamp(default_wfa_end) - wfa_offset).date()
        
        wfa_start_date = st.sidebar.date_input("WFA Début", value=default_wfa_start, key=f"wfa_start_{wfa_train_window}")
        wfa_end_date = st.sidebar.date_input("WFA Fin", value=default_wfa_end, key="wfa_end_today")
        
        if len(tickers) == 0:
            st.warning("👈 Veuillez sélectionner au moins une action dans le menu latéral.")
        elif len(tickers) == 1:
            run_single_mode(tickers[0], period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date)
        else:
            run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date)
            
    elif menu == "🕹️ Paper Trading (Virtuel)":
        page_paper_trading()
    elif menu == "🕹️ Paper Trading (Options)":
        page_options_paper_trading()
    elif menu == "⏪ Backtester Systématique (Options)":
        page_options_backtester(tickers)
    elif menu == "🤖 Paramétrage du Bot (Headless)":
        page_bot_config()
    elif menu == "🧠 Analyse RAG (Gemini)":
        page_rag_gemini(tickers, gemini_api_key)
    elif menu == "🏢 Fondamentaux Financiers":
        page_fundamentals(tickers)
    elif menu == "🧮 Options & Dérivés (Pricing)":
        page_options_pricing(tickers)
    elif menu == "🎓 Académie: Stratégie & Risques":
        page_strategy_academy()
    elif menu == "🎓 Académie: Modélisation Avancée":
        page_advanced_academy()
    elif menu == "🎓 Académie: Options & Dérivés":
        page_options_academy()
    elif menu == "📚 Académie: Indicateurs":
        page_indicators()
    elif menu == "🤖 Académie: XGBoost":
        page_xgboost()
    elif menu == "📰 Académie: News & Sentiment":
        page_news()

if __name__ == "__main__":
    main()
