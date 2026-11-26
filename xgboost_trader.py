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
from tickers_db import (
    MAJOR_STOCKS,
    PREDEFINED_PORTFOLIOS,
    parse_and_validate_watchlist_csv,
    save_translated_watchlist_csv,
)
import glob

# Import du moteur quantitatif centralisé
from quant_core import (
    add_features, get_macro_data as qc_get_macro_data, join_macro_data,
    MLTrader, convert_google_to_yahoo_ticker, get_company_name_from_yahoo,
    get_option_multiplier_and_legislation, get_black_litterman_weights,
    black_scholes, find_strike_for_delta, run_options_backtest_simulation,
    get_portfolio_path, load_portfolio, save_portfolio, reset_portfolio,
    get_options_portfolio_path, load_options_portfolio, save_options_portfolio, reset_options_portfolio,
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
from auto_trader_engine import (
    AutoTraderConfig,
    get_global_auto_trader,
    list_watchlist_csv_files,
    load_tickers_from_csv_files,
    load_auto_trader_config,
    save_auto_trader_config,
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

def run_single_mode(ticker, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None, strategy_cfg=None):
    st.subheader(f"Analyse Individuelle : {get_company_name_from_yahoo(ticker)}")
    model_path = f"models/{ticker}_model.pkl"
    scfg = strategy_cfg or {}
    labeling_mode = scfg.get("labeling_mode", "triple_barrier")
    calibrate_probs = scfg.get("calibrate_probs", True)
    use_trailing_stop = scfg.get("use_trailing_stop", True)
    use_regime_filters = scfg.get("use_regime_filters", True)
    use_vol_sizing = scfg.get("use_vol_sizing", False)
    
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

    df = add_features(df_raw, labeling_mode=labeling_mode)
    if macro_df is not None:
        df = join_macro_data(df, macro_df)
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.button("🧠 Entraîner l'IA sur cette action", use_container_width=True):
            with st.status("Entraînement en cours...") as status:
                try:
                    features = trader.train(
                        df, optimize=optimize_model, use_wfa=use_wfa,
                        wfa_train_window=wfa_train_window, wfa_step=wfa_step,
                        wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date,
                        calibrate=calibrate_probs,
                        use_trailing_stop=use_trailing_stop,
                        use_regime_filters=use_regime_filters,
                        use_vol_sizing=use_vol_sizing
                    )
                    st.session_state[f"features_{ticker}"] = features
                    try:
                        os.makedirs("models", exist_ok=True)
                        trader.save(model_path)
                        get_global_auto_trader().register_models({ticker: trader})
                    except Exception:
                        pass
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
                if st.button("🚀 Exécuter sur le Paper Trading", key=f"exec_paper_buy_{ticker}"):
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
                            pf['positions'][ticker] = {'qty': new_qty, 'avg_price': new_price, 'stop_loss': float(stop_loss), 'take_profit': float(take_profit)}
                        else:
                            pf['positions'][ticker] = {'qty': position_size, 'avg_price': exec_price, 'stop_loss': float(stop_loss), 'take_profit': float(take_profit)}
                        
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
                if st.button("🔴 Liquider Position (Paper Trading)", key=f"exec_paper_sell_{ticker}"):
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
                
                if st.button(f"🚀 Acheter {qty_to_buy} Contrats CALL (Paper Trading Options)", key=f"exec_opt_call_{ticker}"):
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
                            'strike': float(strike_opt),
                            'days_to_expiry': int(t_opt_days),
                            'premium': float(price_opt),
                            'qty': int(qty_to_buy),
                            'multiplier': int(multiplier),
                            'underlying_price_at_buy': float(current_price_opt),
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"CALL Strike {strike_opt}",
                            'qty': int(qty_to_buy),
                            'premium_paid': float(price_opt),
                            'total_cost': float(total_premium)
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
                
                if st.button(f"🔴 Acheter {qty_to_buy} Contrats PUT (Paper Trading Options)", key=f"exec_opt_put_{ticker}"):
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
                            'strike': float(strike_opt),
                            'days_to_expiry': int(t_opt_days),
                            'premium': float(price_opt),
                            'qty': int(qty_to_buy),
                            'multiplier': int(multiplier),
                            'underlying_price_at_buy': float(current_price_opt),
                            'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        pf_opt['history'].append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': 'BUY OPTION',
                            'ticker': ticker,
                            'contract': f"PUT Strike {strike_opt}",
                            'qty': int(qty_to_buy),
                            'premium_paid': float(price_opt),
                            'total_cost': float(total_premium)
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
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("CAGR", f"{adv.get('CAGR', 0)*100:.2f}%", help="Taux de croissance annuel composé de la stratégie.")
            t2.metric("Profit Factor", f"{adv.get('Profit Factor', 0):.2f}", help="Ratio Gains bruts / Pertes brutes (> 1.5 recommandé).")
            t3.metric("Calmar Ratio", f"{adv.get('Calmar Ratio', 0):.2f}", help="CAGR / |Max Drawdown|.")
            t4.metric("Exposition", f"{adv.get('Exposure %', 0)*100:.1f}%", help="Pourcentage du temps passé investi sur le marché.")
            t5.metric("Sharpe Buy & Hold", f"{adv.get('Market Sharpe Ratio', 0):.2f}", help="Sharpe du marché sur la même période de test.")

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
                    # Extraction robuste du modèle XGBoost (VotingClassifier ou CalibratedClassifierCV)
                    if hasattr(trader.model, "estimators_"):
                        fitted_xgb = trader.model.estimators_[0]
                    elif hasattr(trader.model, "calibrated_classifiers_"):
                        fitted_xgb = trader.model.calibrated_classifiers_[0].estimator.estimators_[0]
                    else:
                        fitted_xgb = trader.xgb_model
                    explainer = shap.TreeExplainer(fitted_xgb)
                    shap_features = getattr(trader, "features", features)
                    shap_values = explainer.shap_values(last_row.reindex(columns=shap_features, fill_value=0.0))
                    
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    elif len(shap_values.shape) > 1:
                        sv = shap_values[0]
                    else:
                        sv = shap_values
                    
                    shap_df = pd.DataFrame({'Feature': shap_features, 'Contribution': sv})
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

def run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa=False, wfa_train_window="5Y", wfa_step="6M", wfa_start_date=None, wfa_end_date=None, strategy_cfg=None):
    scfg = strategy_cfg or {}
    labeling_mode = scfg.get("labeling_mode", "triple_barrier")
    calibrate_probs = scfg.get("calibrate_probs", True)
    use_trailing_stop = scfg.get("use_trailing_stop", True)
    use_regime_filters = scfg.get("use_regime_filters", True)
    use_vol_sizing = scfg.get("use_vol_sizing", False)
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
            macro_df = get_macro_data(period, interval)
            views_dict = {}
            for t, df_raw in data_dict.items():
                st.write(f"Entraînement de l'IA pour {t}...")
                df = add_features(df_raw, labeling_mode=labeling_mode)
                if macro_df is not None:
                    df = join_macro_data(df, macro_df)
                trader = MLTrader()
                try:
                    features = trader.train(
                        df, optimize=optimize_model, use_wfa=use_wfa,
                        wfa_train_window=wfa_train_window, wfa_step=wfa_step,
                        wfa_start_date=wfa_start_date, wfa_end_date=wfa_end_date,
                        calibrate=calibrate_probs,
                        use_trailing_stop=use_trailing_stop,
                        use_regime_filters=use_regime_filters,
                        use_vol_sizing=use_vol_sizing
                    )
                except ValueError as ve:
                    st.warning(f"⚠️ {t} : {ve}")
                    # Repli sur entraînement standard
                    features = trader.train(
                        df, optimize=optimize_model, use_wfa=False,
                        calibrate=calibrate_probs,
                        use_trailing_stop=use_trailing_stop,
                        use_regime_filters=use_regime_filters,
                        use_vol_sizing=use_vol_sizing
                    )
                st.session_state[f"port_trader_{t}"] = trader
                try:
                    os.makedirs("models", exist_ok=True)
                    trader.save(f"models/{t}_model.pkl")
                    get_global_auto_trader().register_models({t: trader})
                except Exception:
                    pass
                
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
            returns_df = pd.DataFrame(returns_dict).dropna(how='all').fillna(0.0)
            
            # Application du modèle (annualisation adaptée à l'intervalle)
            ppy = 52 if interval == "1wk" else 252
            bl_weights = get_black_litterman_weights(returns_df, market_weights, views_dict, periods_per_year=ppy)
            
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
                                        df_last = yf.download(t, period="5d", progress=False)
                                        current_price = df_last['Close'].dropna().iloc[-1]
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
                df_p = yf.download(t, period="5d", progress=False)
                current_price = df_p['Close'].dropna().iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
                current_price = float(current_price)
                if pd.isna(current_price) or current_price <= 0:
                    current_price = float(pf['positions'][t]['avg_price'])
            except Exception:
                current_price = float(pf['positions'][t]['avg_price']) # fallback
            
            qty = pf['positions'][t]['qty']
            avg_price = pf['positions'][t]['avg_price']
            value = qty * current_price
            pnl = value - (qty * avg_price)
            pnl_pct = (pnl / (qty * avg_price)) * 100 if (qty * avg_price) > 0 else 0.0
            
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
        
    if st.button("⚠️ Réinitialiser le compte (Remise à 100 000 $)", key="btn_reset_equity_portfolio"):
        reset_portfolio()
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
                close_s = df['Close'].dropna()
                current_price = close_s.iloc[-1].item() if isinstance(close_s.iloc[-1], pd.Series) else float(close_s.iloc[-1])
                if pd.isna(current_price) or current_price <= 0:
                    current_price = float(data['underlying_price_at_buy'])
                sigma = get_implied_volatility(t, current_price)
            except Exception:
                current_price = float(data['underlying_price_at_buy'])
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
        
    if st.button("⚠️ Réinitialiser le compte d'Options", key="btn_reset_options_portfolio"):
        reset_options_portfolio()
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
    
    if st.button("✅ Acheter ce contrat (Paper Trading)", use_container_width=True, key=f"buy_opt_pricing_{ticker}"):
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
                'strike': float(K),
                'days_to_expiry': int(T_days),
                'premium': float(price),
                'qty': int(qty_options),
                'multiplier': int(multiplier),
                'underlying_price_at_buy': float(S_input),
                'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pf_opt['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY OPTION',
                'ticker': ticker,
                'contract': f"{opt_type.upper()} Strike {K}",
                'qty': int(qty_options),
                'premium_paid': float(price),
                'total_cost': float(cout_total)
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
    _sync_custom_watchlists_state()
    pending_bot_pf = st.session_state.pop("pending_pf_bot", None)
    if pending_bot_pf and pending_bot_pf in PREDEFINED_PORTFOLIOS:
        st.session_state["pf_select_bot"] = pending_bot_pf

    col_bot_pf, col_bot_btn = st.columns([6, 1])
    with col_bot_pf:
        st.markdown("**💡 Sélection rapide de portefeuille**")
    with col_bot_btn:
        if st.button("📂", key="btn_import_csv_bot", help="Importer des watchlists CSV depuis un répertoire"):
            if _HAS_ST_DIALOG:
                _watchlist_import_modal("bot")
            else:
                st.session_state["wl_fallback_open_bot"] = not st.session_state.get("wl_fallback_open_bot", False)

    if not _HAS_ST_DIALOG and st.session_state.get("wl_fallback_open_bot", False):
        with st.expander("📂 Importer des Watchlists CSV", expanded=True):
            _render_watchlist_import_body("bot")

    portfolio_choice = st.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0,
        key="pf_select_bot",
        label_visibility="collapsed"
    )

    last_bot_pf = st.session_state.get("last_pf_bot")
    if last_bot_pf is not None and last_bot_pf != portfolio_choice:
        st.session_state["wl_complement_bot"] = []
    st.session_state["last_pf_bot"] = portfolio_choice

    if portfolio_choice != "Sélection Manuelle":
        default_selection = list(PREDEFINED_PORTFOLIOS[portfolio_choice])
    else:
        default_selection = list(config.get("tickers", ["Apple Inc. (US)"]))

    for extra_s in st.session_state.get("wl_complement_bot", []):
        if extra_s not in default_selection:
            default_selection.append(extra_s)

    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    bot_wl_ver = st.session_state.get("wl_version_bot", 0)

    stock_choices = st.multiselect(
        "Rechercher une ou plusieurs actions",
        options=list(MAJOR_STOCKS.keys())[1:],
        default=valid_defaults,
        key=f"bot_stocks_ms_{portfolio_choice}_{bot_wl_ver}"
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

@st.cache_data(ttl=15, show_spinner=False)
def _fetch_live_price_fast(symbol):
    """Récupère le cours en temps réel (cache 15s) pour la supervision live."""
    try:
        tk = yf.Ticker(symbol)
        fi = getattr(tk, "fast_info", None)
        if fi is not None and hasattr(fi, "last_price"):
            lp = fi.last_price
            if lp is not None and float(lp) > 0:
                return float(lp)
        hist = tk.history(period="2d", interval="1m")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            val = float(hist["Close"].dropna().iloc[-1])
            if val > 0:
                return val
        hist_d = tk.history(period="5d")
        if hist_d is not None and not hist_d.empty and "Close" in hist_d.columns:
            val = float(hist_d["Close"].dropna().iloc[-1])
            if val > 0:
                return val
    except Exception:
        return None
    return None


def _extract_technical_snapshot(sym, df, trader=None, prob_buy=None):
    """Extrait un instantané propre des indicateurs techniques clés sur la dernière bougie t."""
    if df is None or df.empty:
        return None
    row = df.iloc[-1]

    def _f(col, mult=1.0, digits=2):
        if col not in df.columns:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        try:
            return round(float(val) * mult, digits)
        except Exception:
            return None

    close_v = _f("Close", 1.0, 2)
    sma200_ratio = _f("SMA_200_Ratio", 100.0, 2)
    rsi_v = _f("RSI", 1.0, 1)
    bb_width = _f("BB_Width", 100.0, 2)

    if sma200_ratio is not None:
        trend_lbl = "🟢 Haussier" if sma200_ratio >= 0 else "🔴 Baissier"
    else:
        trend_lbl = "⚪ N/A"

    if rsi_v is not None:
        if rsi_v > 75:
            rsi_zone = "🔴 Surachat (>75)"
        elif rsi_v < 30:
            rsi_zone = "🟢 Survente (<30)"
        else:
            rsi_zone = "⚪ Neutre"
    else:
        rsi_zone = "⚪ N/A"

    sharpe_v = None
    pf_v = None
    if trader is not None and isinstance(getattr(trader, "advanced_metrics", None), dict):
        s_raw = trader.advanced_metrics.get("Sharpe Ratio")
        p_raw = trader.advanced_metrics.get("Profit Factor")
        if s_raw is not None:
            sharpe_v = round(float(s_raw), 2)
        if p_raw is not None:
            pf_v = round(float(p_raw), 2)

    return {
        "Symbole": sym,
        "Nom": get_company_name_from_yahoo(sym),
        "Cours ($)": close_v,
        "Tendance SMA200": trend_lbl,
        "Écart SMA20 (%)": _f("SMA_20_Ratio", 100.0, 2),
        "Écart SMA50 (%)": _f("SMA_50_Ratio", 100.0, 2),
        "Écart SMA200 (%)": sma200_ratio,
        "RSI (14)": rsi_v,
        "Zone RSI": rsi_zone,
        "MACD Hist": _f("MACD_Hist", 100.0, 3),
        "Bollinger %B": _f("BB_Percent", 100.0, 1),
        "BB Width (%)": bb_width,
        "ADX (14)": _f("ADX", 1.0, 1),
        "DI Diff (+DI/-DI)": _f("DI_Diff", 1.0, 1),
        "Kaufman ER": _f("ER_10", 1.0, 2),
        "NATR (%)": _f("NATR", 100.0, 2),
        "Volume Ratio": _f("Volume_Ratio", 1.0, 2),
        "CMF (20)": _f("CMF_20", 1.0, 2),
        "Donchian 20j (%)": _f("Donchian_Pos_20", 100.0, 1),
        "Dist. 52s (%)": _f("Dist_High_52w", 100.0, 1),
        "RS vs SPY 20j (%)": _f("RS_vs_SPY_20d", 100.0, 2),
        "Prob. Achat (%)": round(float(prob_buy) * 100.0, 1) if prob_buy is not None else None,
        "Sharpe WFA": sharpe_v,
        "PF WFA": pf_v,
    }


def _render_technical_inspector_body(universe_symbols=None, period="5y", interval="1d", labeling_mode="triple_barrier"):
    """Contenu de la fenêtre modale d'inspection technique multi-actions (RSI, SMA, Bollinger...)."""
    top_c1, top_c2, top_c3 = st.columns([3, 1.5, 1.3])
    with top_c1:
        st.caption(
            "Vue synthétique des indicateurs techniques calculés lors de l'apprentissage WFA "
            "(Tendance SMA, Momentum RSI/MACD, Volatilité Bollinger/NATR, Flux CMF/Volume & Score IA)."
        )
    with top_c2:
        if universe_symbols and st.button("🔄 Rafraîchir depuis Yahoo", use_container_width=True, key="tech_insp_refresh"):
            refreshed = []
            prog = st.progress(0.0, text="Actualisation des indicateurs techniques...")
            macro_df = get_macro_data(period, interval)
            for idx_s, sym in enumerate(universe_symbols):
                prog.progress((idx_s + 1) / max(len(universe_symbols), 1), text=f"Calcul indicateurs : {sym}...")
                try:
                    df_raw = yf.download(sym, period=period, interval=interval, progress=False)
                    if df_raw is None or df_raw.empty or len(df_raw) < 30:
                        continue
                    if isinstance(df_raw.columns, pd.MultiIndex):
                        df_raw.columns = df_raw.columns.droplevel(1)
                    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
                    df_ind = add_features(df_raw, labeling_mode=labeling_mode)
                    if macro_df is not None:
                        df_ind = join_macro_data(df_ind, macro_df)
                    tr = st.session_state.get(f"auto_trader_model_{sym}")
                    pb = None
                    if tr is not None and getattr(tr, "is_trained", False):
                        pb = tr.predict(df_ind.iloc[-1:], tr.get_feature_list(df_ind))
                    snap = _extract_technical_snapshot(sym, df_ind, trader=tr, prob_buy=pb)
                    if snap:
                        refreshed.append(snap)
                except Exception:
                    continue
            prog.empty()
            if refreshed:
                st.session_state["auto_training_indicators"] = refreshed
                st.rerun()
    with top_c3:
        if st.button("⬅️ Retour au Trading", type="primary", use_container_width=True, key="tech_insp_back_top"):
            st.session_state["tech_inspector_open"] = False
            st.rerun()

    rows = st.session_state.get("auto_training_indicators", [])
    if not rows:
        st.info("Aucun indicateur en mémoire. Lancez l'apprentissage WFA ou cliquez sur **🔄 Rafraîchir depuis Yahoo** ci-dessus.")
        return

    df_all = pd.DataFrame(rows)

    # KPIs globaux de l'univers analysé
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Actions analysées", len(df_all))
    n_bull = int((df_all["Écart SMA200 (%)"].fillna(-1) >= 0).sum()) if "Écart SMA200 (%)" in df_all else 0
    n_ob = int((df_all["RSI (14)"].fillna(50) > 75).sum()) if "RSI (14)" in df_all else 0
    n_os = int((df_all["RSI (14)"].fillna(50) < 30).sum()) if "RSI (14)" in df_all else 0
    n_sq = int((df_all["BB Width (%)"].fillna(99) < 6.0).sum()) if "BB Width (%)" in df_all else 0
    m2.metric("Haussières (> SMA200)", f"{n_bull} / {len(df_all)}")
    m3.metric("Surachat (RSI > 75)", n_ob)
    m4.metric("Survente (RSI < 30)", n_os)
    m5.metric("Bollinger Squeeze (<6%)", n_sq)

    # Filtres rapides d'expert
    f1, f2, f3, f4 = st.columns([2, 1.3, 1.3, 1.4])
    search_sym = f1.text_input("🔎 Filtrer par symbole ou nom", value="", key="tech_insp_search")
    only_bull = f2.checkbox("Haussier (> SMA200)", value=False, key="tech_insp_only_bull")
    only_safe_rsi = f3.checkbox("RSI ≤ 75 (hors surachat)", value=False, key="tech_insp_safe_rsi")
    only_good_wfa = f4.checkbox("Qualité WFA (PF ≥ 1.0)", value=False, key="tech_insp_good_wfa")

    df_view = df_all.copy()
    if search_sym.strip():
        q = search_sym.strip().lower()
        df_view = df_view[
            df_view["Symbole"].astype(str).str.lower().str.contains(q, na=False)
            | df_view["Nom"].astype(str).str.lower().str.contains(q, na=False)
        ]
    if only_bull and "Écart SMA200 (%)" in df_view:
        df_view = df_view[df_view["Écart SMA200 (%)"].fillna(-999) >= 0]
    if only_safe_rsi and "RSI (14)" in df_view:
        df_view = df_view[df_view["RSI (14)"].fillna(50) <= 75]
    if only_good_wfa and "PF WFA" in df_view:
        df_view = df_view[df_view["PF WFA"].fillna(0) >= 1.0]

    tab_tbl, tab_chart = st.tabs(["📋 Tableau Complet des Indicateurs", "📊 Carte RSI vs Probabilité IA & Bollinger"])

    with tab_tbl:
        if not df_view.empty:
            def _style_rsi(val):
                try:
                    v = float(val)
                    if v > 75:
                        return "color: #ff4b4b; font-weight: 700;"
                    if v < 30:
                        return "color: #00ff88; font-weight: 700;"
                except Exception:
                    pass
                return ""

            def _style_sign(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color: #00ff88;"
                    if v < 0:
                        return "color: #ff4b4b;"
                except Exception:
                    pass
                return ""

            styler = df_view.style
            if hasattr(styler, "map"):
                styler = styler.map(_style_rsi, subset=["RSI (14)"])
                styler = styler.map(_style_sign, subset=["Écart SMA200 (%)", "RS vs SPY 20j (%)", "Sharpe WFA"])
            st.dataframe(styler, use_container_width=True, hide_index=True, height=420)
            st.download_button(
                "📥 Exporter tous les Indicateurs Techniques (CSV)",
                data=df_view.to_csv(index=False).encode("utf-8"),
                file_name="indicateurs_techniques_wfa.csv",
                mime="text/csv",
                key="tech_insp_dl_csv",
            )
        else:
            st.warning("Aucune action ne correspond aux filtres sélectionnés.")

    with tab_chart:
        plot_df = df_view.dropna(subset=["RSI (14)", "Prob. Achat (%)"]).copy()
        if not plot_df.empty:
            color_col = "PF WFA" if "PF WFA" in plot_df.columns and plot_df["PF WFA"].notna().any() else "Écart SMA200 (%)"
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=plot_df["RSI (14)"],
                y=plot_df["Prob. Achat (%)"],
                mode="markers+text",
                text=plot_df["Symbole"],
                textposition="top center",
                marker=dict(
                    size=13,
                    color=plot_df[color_col].fillna(0.0),
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title=color_col),
                    line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "RSI (14): %{x:.1f}<br>"
                    "Prob. Achat IA: %{y:.1f}%<br>"
                    f"{color_col}: %{{marker.color:.2f}}<extra></extra>"
                ),
            ))
            fig_sc.add_vline(x=75, line_dash="dash", line_color="#ff4b4b", annotation_text="Surachat (RSI 75)")
            fig_sc.add_vline(x=30, line_dash="dash", line_color="#00ff88", annotation_text="Survente (RSI 30)")
            fig_sc.add_hline(y=52, line_dash="dot", line_color="#00d2ff", annotation_text="Seuil Achat 52%")
            fig_sc.update_layout(
                title="Carte d'Opportunités : RSI (14) vs Probabilité d'Achat IA (%)",
                xaxis_title="RSI (14)",
                yaxis_title="Probabilité d'Achat IA (%)",
                template="plotly_dark",
                height=420,
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Données insuffisantes pour afficher le graphique RSI vs Probabilité IA.")

    st.divider()
    if st.button("⬅️ Retour au Trading", use_container_width=True, key="tech_insp_back_bottom"):
        st.session_state["tech_inspector_open"] = False
        st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("🔬 Inspecteur Technique Multi-Actions (RSI, SMA, Bollinger...)", width="large")
    def _technical_inspector_modal(universe_symbols=None, period="5y", interval="1d", labeling_mode="triple_barrier"):
        _render_technical_inspector_body(universe_symbols, period, interval, labeling_mode)
else:
    def _technical_inspector_modal(universe_symbols=None, period="5y", interval="1d", labeling_mode="triple_barrier"):
        st.session_state["tech_inspector_open"] = True


# =============================================================================
# MODULE : MODE TRADING AUTOMATIQUE EN BOUCLE (AUTOPILOT)
# =============================================================================
def page_auto_trading(sidebar_tickers=None, sidebar_csv_files=None):
    st.title("⚡ Mode Trading Automatique (Autopilot)")
    st.markdown(
        """
        Pilotez ici le **cycle complet de trading autonome** sans modifier les stratégies existantes :
        1. **Sélectionnez un ou plusieurs fichiers CSV de trackers** (`WatchList/` ou import externe).
        2. **Lancez l'apprentissage** (mode **WFA** ou **Standard**, Optuna, Triple Barrier — identique au Terminal).
        3. **Actionnez le Mode Automatique** : l'application sélectionne les meilleures actions à acheter,
           surveille la rentabilité (Take-Profit, Stop-Loss, Trailing Stop ATR, Breakeven +1R, signal IA),
           vend automatiquement, puis réinvestit en boucle jusqu'à votre interruption.
        """
    )

    engine = get_global_auto_trader()
    saved_cfg = load_auto_trader_config()
    status = engine.get_status()

    # =========================================================================
    # ÉTAPE 1 : SÉLECTION MULTIPLE DES FICHIERS CSV DE TRACKERS
    # =========================================================================
    st.header("1. 📂 Sélection Multiple des Fichiers CSV de Trackers")

    available_csvs = list_watchlist_csv_files()
    default_csvs = [f for f in (sidebar_csv_files or saved_cfg.csv_sources or []) if f in available_csvs]

    col_csv_left, col_csv_right = st.columns([3, 2])
    with col_csv_left:
        selected_csv_files = st.multiselect(
            "Choisir un ou plusieurs fichiers CSV dans WatchList/",
            options=available_csvs,
            default=default_csvs,
            key="auto_csv_multiselect",
            help="Tous les tickers des fichiers CSV sélectionnés sont automatiquement fusionnés et dédupliqués.",
        )

    with col_csv_right:
        uploaded_auto_csvs = st.file_uploader(
            "Ou importer d'autres fichiers CSV externes",
            type=["csv"],
            accept_multiple_files=True,
            key="auto_csv_uploader",
        )
        save_uploaded_to_disk = st.checkbox(
            "Enregistrer les CSV importés dans WatchList/",
            value=True,
            key="auto_csv_save_disk",
        )

    merged_entries = []
    seen_syms = set()

    # 1a. Chargement depuis les CSV sélectionnés dans WatchList/
    if selected_csv_files:
        loaded_bundle = load_tickers_from_csv_files(selected_csv_files)
        for e in loaded_bundle["entries"]:
            if e["symbol"] not in seen_syms:
                seen_syms.add(e["symbol"])
                merged_entries.append(e)
        if loaded_bundle["errors"]:
            for err in loaded_bundle["errors"]:
                st.warning(f"⚠️ {err}")

    # 1b. Chargement depuis les CSV uploadés à la volée
    uploaded_names = []
    if uploaded_auto_csvs:
        for up in uploaded_auto_csvs:
            res = parse_and_validate_watchlist_csv(up, up.name)
            if res["is_valid"]:
                saved_name = up.name
                if save_uploaded_to_disk:
                    ok_save, _ = save_translated_watchlist_csv(res["translated_csv"], res["clean_filename"])
                    if ok_save:
                        saved_name = res["clean_filename"]
                if saved_name not in uploaded_names:
                    uploaded_names.append(saved_name)
                for item in res["entries"]:
                    sym = convert_google_to_yahoo_ticker(item["symbol"])
                    if sym and sym not in seen_syms:
                        seen_syms.add(sym)
                        merged_entries.append({
                            "symbol": sym,
                            "name": item.get("name", sym),
                            "display_name": item.get("display_name", sym),
                            "exchange": item.get("exchange", ""),
                            "source_file": up.name,
                        })
            else:
                st.error(f"❌ {up.name} : {' ; '.join(res.get('errors', ['Format invalide']))}")

    # 1c. Complément optionnel avec les tickers déjà choisis dans la barre latérale
    include_sidebar = st.checkbox(
        f"Inclure également les tickers du menu latéral ({len(sidebar_tickers or [])} titre(s))",
        value=(len(merged_entries) == 0 and bool(sidebar_tickers)),
        key="auto_include_sidebar_tickers",
    )
    if include_sidebar and sidebar_tickers:
        for t in sidebar_tickers:
            sym = convert_google_to_yahoo_ticker(t)
            if sym and sym not in seen_syms:
                seen_syms.add(sym)
                merged_entries.append({
                    "symbol": sym,
                    "name": get_company_name_from_yahoo(sym),
                    "display_name": f"{get_company_name_from_yahoo(sym)} ({sym})",
                    "exchange": "",
                    "source_file": "Menu latéral",
                })

    universe_symbols = [e["symbol"] for e in merged_entries]
    all_csv_sources = list(selected_csv_files) + uploaded_names

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    c_stat1.metric("Fichiers CSV sélectionnés", len(all_csv_sources))
    c_stat2.metric("Trackers uniques fusionnés", len(universe_symbols))

    # Détecter combien de modèles sont déjà entraînés (en mémoire ou sur disque)
    ready_symbols = []
    for sym in universe_symbols:
        in_mem = sym in status.get("trained_symbols", [])
        in_ss = (
            (f"trader_{sym}" in st.session_state and getattr(st.session_state[f"trader_{sym}"], "is_trained", False))
            or (f"port_trader_{sym}" in st.session_state and getattr(st.session_state[f"port_trader_{sym}"], "is_trained", False))
            or (f"auto_trader_model_{sym}" in st.session_state and getattr(st.session_state[f"auto_trader_model_{sym}"], "is_trained", False))
        )
        on_disk = os.path.exists(os.path.join("models", f"{sym}_model.pkl"))
        if in_mem or in_ss or on_disk:
            ready_symbols.append(sym)
    c_stat3.metric("Modèles entraînés prêts", f"{len(ready_symbols)} / {len(universe_symbols)}")

    if merged_entries:
        with st.expander(f"👁️ Aperçu des {len(merged_entries)} trackers sélectionnés", expanded=False):
            st.dataframe(
                pd.DataFrame(merged_entries)[["symbol", "name", "exchange", "source_file"]].rename(
                    columns={"symbol": "Symbole Yahoo", "name": "Nom", "exchange": "Bourse", "source_file": "Fichier CSV"}
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("👆 Sélectionnez un ou plusieurs fichiers CSV de trackers ci-dessus pour constituer votre univers d'apprentissage et de trading automatique.")

    st.divider()

    # =========================================================================
    # ÉTAPE 2 : APPRENTISSAGE (IDENTIQUE AU TERMINAL — WFA OU STANDARD)
    # =========================================================================
    st.header("2. 🧠 Apprentissage des Modèles (WFA ou Standard)")
    st.caption("Le moteur d'apprentissage et les stratégies (`quant_core.py`) restent strictement identiques.")

    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    auto_period = col_l1.selectbox("Période d'historique", ["2y", "5y", "10y", "max"], index=1, key="auto_period")
    auto_interval = col_l2.selectbox("Intervalle", ["1d", "1wk"], index=0, key="auto_interval")
    auto_learning_mode = col_l3.selectbox(
        "Mode d'apprentissage",
        ["Standard (Train/Test)", "Walk-Forward Analysis (WFA)"],
        index=1 if saved_cfg.learning_mode.startswith("Walk-Forward") else 0,
        key="auto_learning_mode",
    )
    auto_optimize = col_l4.checkbox("🧠 Auto-Optimisation (Optuna)", value=False, key="auto_optimize")

    auto_use_wfa = auto_learning_mode.startswith("Walk-Forward")
    auto_wfa_window = "5Y"
    auto_wfa_step = "6M"
    auto_wfa_start = None
    auto_wfa_end = None

    if auto_use_wfa:
        wc1, wc2, wc3, wc4 = st.columns(4)
        auto_wfa_window = wc1.selectbox("Fenêtre WFA", ["2Y", "5Y", "7Y"], index=1, key="auto_wfa_window")
        auto_wfa_step = wc2.selectbox("Pas d'avancement WFA", ["1M", "2M", "6M", "1Y"], index=2, key="auto_wfa_step")
        win_offsets = {"2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5), "7Y": pd.DateOffset(years=7)}
        default_end = datetime.now().date()
        default_start = (pd.Timestamp(default_end) - win_offsets[auto_wfa_window]).date()
        auto_wfa_start = wc3.date_input("WFA Début", value=default_start, key=f"auto_wfa_start_{auto_wfa_window}")
        auto_wfa_end = wc4.date_input("WFA Fin", value=default_end, key="auto_wfa_end")

    with st.expander("⚙️ Options Stratégie d'Apprentissage (Identiques au Terminal)", expanded=False):
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        auto_lbl = sc1.selectbox("Cible ML", ["Triple Barrier (ATR 5j)", "Classique (J+1)"], index=0, key="auto_lbl")
        auto_cal = sc2.checkbox("Calibration Probs", value=True, key="auto_cal")
        auto_tsl_train = sc3.checkbox("Trailing Stop Backtest", value=True, key="auto_tsl_train")
        auto_reg_train = sc4.checkbox("Filtres SMA200 & RSI", value=True, key="auto_reg_train")
        auto_vol_train = sc5.checkbox("Vol Sizing ATR", value=False, key="auto_vol_train")

    auto_labeling_mode = "triple_barrier" if auto_lbl.startswith("Triple") else "next_day"

    col_tr_btn1, col_tr_btn2 = st.columns([3, 1])
    with col_tr_btn1:
        launch_training = st.button(
            f"🧠 Lancer l'Apprentissage ({auto_learning_mode}) sur {len(universe_symbols)} tracker(s)",
            type="primary",
            use_container_width=True,
            disabled=(len(universe_symbols) == 0 or status["is_running"]),
            key="btn_auto_train_all",
        )
    with col_tr_btn2:
        skip_already_trained = st.checkbox(
            "Ignorer modèles déjà sur disque",
            value=False,
            key="auto_skip_existing",
            help="Si coché, réutilise les modèles déjà sauvegardés dans models/ sans les ré-entraîner.",
        )

    if launch_training and universe_symbols:
        os.makedirs("models", exist_ok=True)
        trained_summary = []
        indicators_snapshots = []
        newly_trained = {}
        progress_bar = st.progress(0.0, text="Initialisation de l'apprentissage...")
        macro_df = get_macro_data(auto_period, auto_interval)

        for idx, sym in enumerate(universe_symbols):
            progress_bar.progress(
                idx / max(len(universe_symbols), 1),
                text=f"Apprentissage {idx + 1}/{len(universe_symbols)} : {sym} ({auto_learning_mode})...",
            )
            model_path = os.path.join("models", f"{sym}_model.pkl")

            if skip_already_trained and os.path.exists(model_path):
                try:
                    loaded_m = MLTrader.load(model_path)
                    if loaded_m and getattr(loaded_m, "is_trained", False):
                        st.session_state[f"auto_trader_model_{sym}"] = loaded_m
                        newly_trained[sym] = loaded_m
                        trained_summary.append({
                            "Symbole": sym,
                            "Nom": get_company_name_from_yahoo(sym),
                            "Mode": "Chargé (disque)",
                            "Précision": f"{loaded_m.accuracy:.1%}",
                            "Sharpe": round(float(loaded_m.advanced_metrics.get("Sharpe Ratio", 0.0) or 0.0), 2),
                            "Profit Factor": round(float(loaded_m.advanced_metrics.get("Profit Factor", 0.0) or 0.0), 2),
                            "Statut": "✅ Prêt",
                        })
                        continue
                except Exception:
                    pass

            try:
                df_raw = yf.download(sym, period=auto_period, interval=auto_interval, progress=False)
                if df_raw is None or df_raw.empty or len(df_raw) < 60:
                    trained_summary.append({
                        "Symbole": sym, "Nom": get_company_name_from_yahoo(sym),
                        "Mode": auto_learning_mode, "Précision": "-", "Sharpe": "-", "Profit Factor": "-",
                        "Statut": "⚠️ Données insuffisantes",
                    })
                    continue

                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.droplevel(1)
                df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

                df = add_features(df_raw, labeling_mode=auto_labeling_mode)
                if macro_df is not None:
                    df = join_macro_data(df, macro_df)

                trader = MLTrader()
                mode_used = auto_learning_mode
                try:
                    features = trader.train(
                        df,
                        optimize=auto_optimize,
                        use_wfa=auto_use_wfa,
                        wfa_train_window=auto_wfa_window,
                        wfa_step=auto_wfa_step,
                        wfa_start_date=auto_wfa_start,
                        wfa_end_date=auto_wfa_end,
                        calibrate=auto_cal,
                        use_trailing_stop=auto_tsl_train,
                        use_regime_filters=auto_reg_train,
                        use_vol_sizing=auto_vol_train,
                    )
                except ValueError:
                    mode_used = "Standard (repli WFA)"
                    features = trader.train(
                        df,
                        optimize=auto_optimize,
                        use_wfa=False,
                        calibrate=auto_cal,
                        use_trailing_stop=auto_tsl_train,
                        use_regime_filters=auto_reg_train,
                        use_vol_sizing=auto_vol_train,
                    )

                trader.save(model_path)
                st.session_state[f"auto_trader_model_{sym}"] = trader
                newly_trained[sym] = trader

                last_prob = trader.predict(df.iloc[-1:], features)
                snap_ind = _extract_technical_snapshot(sym, df, trader=trader, prob_buy=last_prob)
                if snap_ind is not None:
                    indicators_snapshots.append(snap_ind)
                trained_summary.append({
                    "Symbole": sym,
                    "Nom": get_company_name_from_yahoo(sym),
                    "Mode": mode_used,
                    "Précision": f"{trader.accuracy:.1%}",
                    "Sharpe": round(float(trader.advanced_metrics.get("Sharpe Ratio", 0.0) or 0.0), 2),
                    "Profit Factor": round(float(trader.advanced_metrics.get("Profit Factor", 0.0) or 0.0), 2),
                    "Prob. Achat": f"{last_prob*100:.1f}%" if last_prob is not None else "-",
                    "Statut": "✅ Entraîné",
                })
            except Exception as ex:
                trained_summary.append({
                    "Symbole": sym, "Nom": get_company_name_from_yahoo(sym),
                    "Mode": auto_learning_mode, "Précision": "-", "Sharpe": "-", "Profit Factor": "-",
                    "Statut": f"❌ Erreur ({str(ex)[:40]})",
                })

        progress_bar.progress(1.0, text="✅ Apprentissage terminé !")
        if newly_trained:
            engine.register_models(newly_trained)
            st.session_state["auto_training_summary"] = trained_summary
            if indicators_snapshots:
                st.session_state["auto_training_indicators"] = indicators_snapshots
            st.success(f"✅ {len(newly_trained)}/{len(universe_symbols)} modèle(s) entraîné(s) et prêt(s) pour le Mode Automatique !")
            ready_symbols = sorted(set(ready_symbols).union(newly_trained.keys()))
        else:
            st.error("Aucun modèle n'a pu être entraîné. Vérifiez votre connexion ou les tickers sélectionnés.")

    if st.session_state.get("auto_training_summary"):
        with st.expander("📊 Résultats du dernier apprentissage", expanded=True):
            st.dataframe(pd.DataFrame(st.session_state["auto_training_summary"]), use_container_width=True, hide_index=True)

    # Bouton dédié d'ouverture de la fenêtre modale d'inspection technique (RSI, SMA, Bollinger...)
    if st.session_state.get("auto_training_indicators") or len(ready_symbols) > 0:
        if st.button(
            "🔬 Visualiser les Indicateurs Techniques (RSI, SMA, Bollinger, ADX, MACD...)",
            use_container_width=True,
            key="btn_open_tech_inspector",
            help="Ouvre une fenêtre dédiée avec tous les indicateurs techniques des actions entraînées et un bouton Retour au Trading.",
        ):
            if hasattr(st, "dialog"):
                _technical_inspector_modal(universe_symbols, auto_period, auto_interval, auto_labeling_mode)
            else:
                st.session_state["tech_inspector_open"] = True

    if not hasattr(st, "dialog") and st.session_state.get("tech_inspector_open", False):
        with st.expander("🔬 Inspecteur Technique Multi-Actions (RSI, SMA, Bollinger...)", expanded=True):
            _render_technical_inspector_body(universe_symbols, auto_period, auto_interval, auto_labeling_mode)

    st.divider()

    # =========================================================================
    # ÉTAPE 3 : PILOTAGE DU MODE AUTOMATIQUE EN BOUCLE + COMMANDES EXPERT
    # =========================================================================
    st.header("3. ⚡ Pilotage du Mode Automatique en Boucle")

    # Synchroniser les modèles de la session Streamlit vers le moteur
    session_models = {}
    for sym in universe_symbols:
        for prefix in ("auto_trader_model_", "port_trader_", "trader_"):
            m = st.session_state.get(f"{prefix}{sym}")
            if m is not None and getattr(m, "is_trained", False):
                session_models[sym] = m
                break
    if session_models:
        engine.register_models(session_models)

    # Panneau des paramètres Expert
    with st.expander("🎛️ Commandes Expert (Rentabilité, Risque & Protections)", expanded=not status["is_running"]):
        p1, p2, p3, p4 = st.columns(4)
        scan_interval_sec = p1.number_input(
            "Intervalle de boucle (sec)", min_value=10, max_value=86400,
            value=int(saved_cfg.scan_interval_sec), step=10, key="auto_cfg_interval",
            help="Délai entre deux cycles complets (Vente -> Sélection -> Achat).",
        )
        max_open_positions = p2.number_input(
            "Positions max simultanées", min_value=1, max_value=30,
            value=int(saved_cfg.max_open_positions), step=1, key="auto_cfg_max_pos",
        )
        position_size_pct = p3.slider(
            "Allocation par position (%)", min_value=2.0, max_value=50.0,
            value=float(saved_cfg.position_size_pct), step=1.0, key="auto_cfg_pos_size",
        )
        min_cash_reserve_pct = p4.slider(
            "Réserve Cash minimale (%)", min_value=0.0, max_value=50.0,
            value=float(saved_cfg.min_cash_reserve_pct), step=5.0, key="auto_cfg_cash_res",
        )

        r1, r2, r3, r4 = st.columns(4)
        min_buy_prob_pct = r1.slider(
            "Seuil Conviction IA Achat (%)", min_value=50, max_value=85,
            value=int(round(saved_cfg.min_buy_prob * 100)), step=1, key="auto_cfg_buy_prob",
        )
        sell_prob_pct = r2.slider(
            "Seuil Signal IA Vente (%)", min_value=20, max_value=50,
            value=int(round(saved_cfg.sell_prob_threshold * 100)), step=1, key="auto_cfg_sell_prob",
        )
        take_profit_pct = r3.number_input(
            "Take-Profit Rentabilité (%)", min_value=0.0, max_value=100.0,
            value=float(saved_cfg.take_profit_pct), step=0.5, key="auto_cfg_tp_pct",
            help="Vente automatique dès que le gain atteint ce % (0 = uniquement TP ATR).",
        )
        stop_loss_pct = r4.number_input(
            "Stop-Loss Max Perte (%)", min_value=0.0, max_value=50.0,
            value=float(saved_cfg.stop_loss_pct), step=0.5, key="auto_cfg_sl_pct",
            help="Vente automatique de protection si la perte atteint ce % (0 = uniquement SL ATR).",
        )

        s1, s2, s3, s4 = st.columns(4)
        use_atr_stops = s1.checkbox("Stops ATR Dynamiques", value=bool(saved_cfg.use_atr_stops), key="auto_cfg_atr")
        use_trailing_stop = s2.checkbox("Trailing Stop ATR", value=bool(saved_cfg.use_trailing_stop), key="auto_cfg_tsl")
        use_breakeven_stop = s3.checkbox("Breakeven Auto (+1R)", value=bool(saved_cfg.use_breakeven_stop), key="auto_cfg_be")
        trailing_atr_mult = s4.number_input(
            "Multiplicateur Trailing ATR", min_value=0.5, max_value=5.0,
            value=float(saved_cfg.trailing_atr_mult), step=0.25, key="auto_cfg_tsl_mult",
        )

        g1, g2, g3, g4, g5 = st.columns(5)
        use_vix_breaker = g1.checkbox("Coupe-circuit VIX", value=bool(saved_cfg.use_vix_breaker), key="auto_cfg_vix")
        vix_max_threshold = g2.number_input("Seuil VIX Max", min_value=15.0, max_value=60.0, value=float(saved_cfg.vix_max_threshold), step=1.0, key="auto_cfg_vix_val")
        use_rsi_shield = g3.checkbox("Bouclier RSI (>75)", value=bool(saved_cfg.use_rsi_shield), key="auto_cfg_rsi")
        use_sma200_filter = g4.checkbox("Filtre Tendance SMA200", value=bool(saved_cfg.use_sma200_filter), key="auto_cfg_sma200")
        dry_run = g5.checkbox("🧪 Mode Dry-Run (Simulation)", value=bool(saved_cfg.dry_run), key="auto_cfg_dry_run", help="Enregistre les ordres sans modifier portfolio.json.")

        x1, x2 = st.columns(2)
        max_drawdown_pct = x1.slider(
            "Coupe-circuit Drawdown Max (%)", min_value=2.0, max_value=50.0,
            value=float(saved_cfg.max_drawdown_pct), step=1.0, key="auto_cfg_dd",
            help="Bloque automatiquement tout nouvel achat si le portefeuille perd ce % depuis son pic.",
        )
        cooldown_sec = x2.number_input(
            "Cooldown anti-whipsaw après vente (sec)", min_value=0, max_value=86400,
            value=int(saved_cfg.cooldown_sec), step=30, key="auto_cfg_cooldown",
        )

        st.markdown("**🎯 Qualité WFA & Sizing Adaptatif (Régime SPY / VIX)**")
        q1, q2, q3, q4, q5 = st.columns(5)
        use_wfa_quality_filter = q1.checkbox(
            "Filtre Qualité WFA",
            value=bool(saved_cfg.use_wfa_quality_filter),
            key="auto_cfg_wfa_qual",
            help="N'achète que les actions dont le modèle a été rentable en apprentissage (WFA).",
        )
        min_wfa_profit_factor = q2.number_input(
            "Profit Factor Min", min_value=0.0, max_value=5.0,
            value=float(saved_cfg.min_wfa_profit_factor), step=0.05, key="auto_cfg_min_pf",
        )
        min_wfa_sharpe = q3.number_input(
            "Sharpe WFA Min", min_value=-2.0, max_value=3.0,
            value=float(saved_cfg.min_wfa_sharpe), step=0.05, key="auto_cfg_min_sharpe",
        )
        use_adaptive_regime_sizing = q4.checkbox(
            "Sizing Adaptatif SPY/VIX",
            value=bool(saved_cfg.use_adaptive_regime_sizing),
            key="auto_cfg_adapt_size",
            help="Réduit automatiquement la taille des positions si VIX >= 20 ou SPY < SMA200.",
        )
        defensive_size_multiplier = q5.slider(
            "Facteur Défensif", min_value=0.25, max_value=1.0,
            value=float(saved_cfg.defensive_size_multiplier), step=0.05, key="auto_cfg_def_mult",
            help="Ex: 0.50 divise par 2 la taille des achats en marché nerveux (15% -> 7.5%).",
        )

    current_cfg = AutoTraderConfig(
        tickers=universe_symbols,
        csv_sources=all_csv_sources,
        scan_interval_sec=int(scan_interval_sec),
        max_open_positions=int(max_open_positions),
        position_size_pct=float(position_size_pct),
        min_cash_reserve_pct=float(min_cash_reserve_pct),
        min_buy_prob=float(min_buy_prob_pct) / 100.0,
        sell_prob_threshold=float(sell_prob_pct) / 100.0,
        take_profit_pct=float(take_profit_pct),
        stop_loss_pct=float(stop_loss_pct),
        use_atr_stops=bool(use_atr_stops),
        use_trailing_stop=bool(use_trailing_stop),
        trailing_atr_mult=float(trailing_atr_mult),
        use_breakeven_stop=bool(use_breakeven_stop),
        use_vix_breaker=bool(use_vix_breaker),
        vix_max_threshold=float(vix_max_threshold),
        use_rsi_shield=bool(use_rsi_shield),
        use_sma200_filter=bool(use_sma200_filter),
        use_wfa_quality_filter=bool(use_wfa_quality_filter),
        min_wfa_profit_factor=float(min_wfa_profit_factor),
        min_wfa_sharpe=float(min_wfa_sharpe),
        use_adaptive_regime_sizing=bool(use_adaptive_regime_sizing),
        defensive_size_multiplier=float(defensive_size_multiplier),
        max_drawdown_pct=float(max_drawdown_pct),
        cooldown_sec=int(cooldown_sec),
        dry_run=bool(dry_run),
        period=auto_period,
        interval=auto_interval,
        labeling_mode=auto_labeling_mode,
        learning_mode=auto_learning_mode,
    )

    # Barre de contrôle principale (Start / Stop / Pause / Cycle unique / Kill Switch)
    b1, b2, b3, b4, b5 = st.columns([2.2, 2.2, 1.3, 1.5, 1.3])
    with b1:
        if not status["is_running"]:
            if st.button(
                "🚀 ACTIVER LE MODE AUTOMATIQUE",
                type="primary",
                use_container_width=True,
                disabled=(len(ready_symbols) == 0),
                key="btn_start_autopilot",
            ):
                save_auto_trader_config(current_cfg)
                ok, msg = engine.start(current_cfg, trained_models=session_models)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.success("🟢 **AUTOPILOT ACTIF (EN BOUCLE)**" if not status["is_paused"] else "⏸️ **AUTOPILOT EN PAUSE**")

    with b2:
        if st.button(
            "🛑 INTERROMPRE LE MODE AUTOMATIQUE",
            use_container_width=True,
            disabled=(not status["is_running"]),
            key="btn_stop_autopilot",
        ):
            ok, msg = engine.stop()
            st.warning(msg)
            st.rerun()

    with b3:
        if status["is_running"]:
            if status["is_paused"]:
                if st.button("▶️ Reprendre", use_container_width=True, key="btn_resume_autopilot"):
                    engine.resume()
                    st.rerun()
            else:
                if st.button("⏸️ Pause", use_container_width=True, key="btn_pause_autopilot"):
                    engine.pause()
                    st.rerun()
        else:
            if st.button("💾 Sauver Config", use_container_width=True, key="btn_save_autopilot_cfg"):
                save_auto_trader_config(current_cfg)
                st.toast("Configuration Autopilot sauvegardée !", icon="💾")

    with b4:
        if st.button(
            "⚡ Exécuter 1 Cycle",
            use_container_width=True,
            disabled=(len(ready_symbols) == 0 or status["is_running"]),
            key="btn_single_cycle_autopilot",
            help="Lance un cycle unique (Vente -> Sélection -> Achat) pour tester la configuration.",
        ):
            save_auto_trader_config(current_cfg)
            engine.config = current_cfg
            engine.register_models(session_models)
            with st.spinner("Exécution d'un cycle complet d'analyse et d'ordres..."):
                report = engine.run_single_cycle()
            st.success(
                f"Cycle #{report['cycle']} terminé : {len(report['sells'])} vente(s), "
                f"{len(report['buys'])} achat(s) | Équité : {report['equity']:,.2f} $"
            )
            st.rerun()

    with b5:
        if st.button("🔄 Actualiser", use_container_width=True, key="btn_refresh_autopilot"):
            st.rerun()

    # Commande Expert : Kill Switch (Liquidation Totale)
    with st.expander("🚨 Commandes d'Urgence (Kill Switch)", expanded=False):
        st.warning("La liquidation totale interrompt immédiatement la boucle automatique et vend toutes les positions ouvertes dans `portfolio.json` pour repasser 100% Cash.")
        confirm_liq = st.checkbox("Je confirme vouloir tout vendre et passer 100% Cash", key="confirm_kill_switch")
        if st.button("🚨 LIQUIDATION TOTALE (100% CASH)", disabled=not confirm_liq, key="btn_kill_switch"):
            closed, pnl_liq, liq_msg = engine.emergency_liquidate_all()
            st.warning(liq_msg)
            st.rerun()

    if len(ready_symbols) == 0 and universe_symbols:
        st.warning("⚠️ Lancez d'abord l'**Étape 2 (Apprentissage)** pour entraîner vos modèles avant d'activer le Mode Automatique.")

    # =========================================================================
    # TABLEAU DE BORD TEMPS RÉEL DU MODE AUTOMATIQUE
    # =========================================================================
    status = engine.get_status()
    pf = load_portfolio()

    st.divider()
    live_ts = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <style>
        @keyframes pulse-live {{
            0% {{ box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.75); }}
            70% {{ box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }}
        }}
        .live-badge {{
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(0, 255, 136, 0.12); border: 1px solid #00ff88;
            color: #00ff88; padding: 3px 10px; border-radius: 999px;
            font-size: 0.80rem; font-weight: 700; letter-spacing: 0.5px;
        }}
        .live-dot {{
            width: 9px; height: 9px; border-radius: 50%;
            background: #00ff88; display: inline-block;
            animation: pulse-live 1.6s infinite;
        }}
        </style>
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
            <h3 style="margin:0;">📡 Supervision en Temps Réel</h3>
            <span class="live-badge"><span class="live-dot"></span> LIVE • Cotations actualisées à {live_ts}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calcul temps réel des positions ouvertes (Cours Live, PRU, SL, TP, Gain/Perte)
    positions = pf.get("positions", {})
    live_pos_rows = []
    total_positions_value = 0.0
    total_invested_cost = 0.0
    total_unrealized_pnl = 0.0

    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0))
        if qty <= 0:
            continue
        pru = float(p.get("avg_price", 0.0))
        sl_val = float(p["stop_loss"]) if p.get("stop_loss") else None
        tp_val = float(p["take_profit"]) if p.get("take_profit") else None
        live_px = _fetch_live_price_fast(sym)
        if live_px is None or live_px <= 0:
            live_px = pru

        pos_val = qty * live_px
        cost_basis = qty * pru
        unreal_pnl = pos_val - cost_basis
        unreal_pct = ((live_px - pru) / pru * 100.0) if pru > 0 else 0.0

        total_positions_value += pos_val
        total_invested_cost += cost_basis
        total_unrealized_pnl += unreal_pnl

        dist_sl_pct = ((live_px - sl_val) / live_px * 100.0) if (sl_val and live_px > 0) else None
        dist_tp_pct = ((tp_val - live_px) / live_px * 100.0) if (tp_val and live_px > 0) else None

        live_pos_rows.append({
            "Symbole": sym,
            "Nom": get_company_name_from_yahoo(sym),
            "Quantité": round(qty, 4),
            "PRU ($)": round(pru, 2),
            "Cours Live ($)": round(live_px, 2),
            "Stop-Loss ($)": round(sl_val, 2) if sl_val is not None else None,
            "Take-Profit ($)": round(tp_val, 2) if tp_val is not None else None,
            "Valeur Live ($)": round(pos_val, 2),
            "Gain/Perte ($)": round(unreal_pnl, 2),
            "Gain/Perte (%)": round(unreal_pct, 2),
            "Marge SL (%)": round(dist_sl_pct, 2) if dist_sl_pct is not None else None,
            "Cible TP (%)": round(dist_tp_pct, 2) if dist_tp_pct is not None else None,
        })

    # Calcul des gains/pertes réalisés sur les ventes (historique portefeuille)
    pf_history = pf.get("history", []) or []
    sell_records = [h for h in pf_history if "SELL" in str(h.get("action", "")).upper()]
    buy_records = [h for h in pf_history if "BUY" in str(h.get("action", "")).upper()]
    total_realized_pnl_pf = sum(float(h.get("pnl", 0.0) or 0.0) for h in sell_records)
    live_equity = float(pf.get("cash", 0.0)) + total_positions_value
    total_net_pnl = total_realized_pnl_pf + total_unrealized_pnl
    unreal_total_pct = (total_unrealized_pnl / total_invested_cost * 100.0) if total_invested_cost > 0 else 0.0

    # Bandeau KPI Temps Réel (2 rangées claires)
    k1, k2, k3, k4 = st.columns(4)
    state_label = "🟢 ACTIF" if (status["is_running"] and not status["is_paused"]) else ("⏸️ PAUSE" if status["is_paused"] else "🔴 ARRÊTÉ")
    k1.metric("Statut Moteur", state_label, delta=f"Cycle #{status['cycle_count']}")
    k2.metric("Cash Disponible", f"{pf.get('cash', 0.0):,.2f} $")
    k3.metric("Équité Live Totale", f"{live_equity:,.2f} $", delta=f"{len(live_pos_rows)} position(s)")
    k4.metric("Ordres (Achats / Ventes)", f"{len(buy_records)} / {len(sell_records)}", delta=f"Session auto: {status['total_buys']}/{status['total_sells']}")

    p1_kpi, p2_kpi, p3_kpi, p4_kpi = st.columns(4)
    p1_kpi.metric("Gain / Perte Latent (Live)", f"{total_unrealized_pnl:+,.2f} $", delta=f"{unreal_total_pct:+.2f}%")
    p2_kpi.metric("Gain / Perte Réalisé (Ventes)", f"{total_realized_pnl_pf:+,.2f} $", delta=f"Auto: {status['realized_pnl']:+,.2f} $")
    p3_kpi.metric("Résultat Total Net (Gains/Pertes)", f"{total_net_pnl:+,.2f} $", delta="Latent + Réalisé", delta_color="normal" if total_net_pnl >= 0 else "inverse")
    win_sells = sum(1 for h in sell_records if float(h.get("pnl", 0.0) or 0.0) > 0)
    win_rate_pct = (win_sells / len(sell_records) * 100.0) if sell_records else 0.0
    p4_kpi.metric("Taux de Ventes Gagnantes", f"{win_rate_pct:.1f}%", delta=f"{win_sells}/{len(sell_records)} ventes")

    if status.get("drawdown_halted"):
        st.error(f"🚨 Coupe-circuit Drawdown actif ({status['drawdown_pct']:.2f}%) : nouveaux achats suspendus.")
    if status.get("market_regime"):
        st.caption(
            f"🏛️ **Régime de Marché (SPY/VIX)** : `{status['market_regime']}` • "
            f"**Allocation effective / position** : `{status.get('effective_position_pct', position_size_pct):.1f}%`"
        )
    if status.get("last_error"):
        st.warning(f"⚠️ Dernière erreur moteur : {status['last_error']}")

    tab_pos, tab_anim, tab_rank, tab_trades, tab_logs = st.tabs([
        "📊 Positions Live (Cours, PRU, SL, TP, PnL)",
        "🎬 Animation Graphique Live & Gains/Pertes",
        "🏆 Classement IA (Dernier Scan)",
        "📜 Historique Achats & Ventes",
        "🖥️ Journal d'Exécution (Logs)",
    ])

    with tab_pos:
        if live_pos_rows:
            df_live = pd.DataFrame(live_pos_rows)

            def _color_pnl(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color: #00ff88; font-weight: 700;"
                    if v < 0:
                        return "color: #ff4b4b; font-weight: 700;"
                except Exception:
                    pass
                return ""

            styled_live = df_live.style.map(_color_pnl, subset=["Gain/Perte ($)", "Gain/Perte (%)"])
            st.dataframe(styled_live, use_container_width=True, hide_index=True)

            # Graphique comparatif Cours Live vs PRU / Stop-Loss / Take-Profit
            fig_levels = go.Figure()
            syms_lbl = [r["Symbole"] for r in live_pos_rows]
            fig_levels.add_trace(go.Bar(
                name="PRU ($)", x=syms_lbl, y=[r["PRU ($)"] for r in live_pos_rows],
                marker_color="rgba(148, 163, 184, 0.55)",
            ))
            fig_levels.add_trace(go.Bar(
                name="Cours Live ($)", x=syms_lbl, y=[r["Cours Live ($)"] for r in live_pos_rows],
                marker_color=["#00ff88" if r["Gain/Perte ($)"] >= 0 else "#ff4b4b" for r in live_pos_rows],
                text=[f"{r['Cours Live ($)']:.2f} $ ({r['Gain/Perte (%)']:+.1f}%)" for r in live_pos_rows],
                textposition="outside",
            ))
            sl_vals = [r["Stop-Loss ($)"] for r in live_pos_rows]
            tp_vals = [r["Take-Profit ($)"] for r in live_pos_rows]
            if any(v is not None for v in sl_vals):
                fig_levels.add_trace(go.Scatter(
                    name="🛑 Stop-Loss ($)", x=syms_lbl, y=sl_vals,
                    mode="markers", marker=dict(symbol="triangle-down", size=14, color="#ff4b4b", line=dict(color="white", width=1)),
                ))
            if any(v is not None for v in tp_vals):
                fig_levels.add_trace(go.Scatter(
                    name="🎯 Take-Profit ($)", x=syms_lbl, y=tp_vals,
                    mode="markers", marker=dict(symbol="triangle-up", size=14, color="#00ff88", line=dict(color="white", width=1)),
                ))
            fig_levels.update_layout(
                title="Cours Temps Réel vs PRU, Stop-Loss & Take-Profit",
                barmode="group", template="plotly_dark", height=380,
                yaxis_title="Prix ($)", legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_levels, use_container_width=True)
        else:
            st.info("Aucune position ouverte actuellement dans le portefeuille.")

        eq_hist = status.get("equity_history", [])
        if len(eq_hist) >= 2:
            df_eq = pd.DataFrame(eq_hist)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df_eq["time"], y=df_eq["equity"], mode="lines+markers", name="Équité Totale ($)", line=dict(color="#00ff88", width=2)))
            fig_eq.add_trace(go.Scatter(x=df_eq["time"], y=df_eq["cash"], mode="lines", name="Cash ($)", line=dict(color="#00d2ff", dash="dot")))
            fig_eq.update_layout(title="Courbe d'Équité de la Session Automatique", template="plotly_dark", height=330)
            st.plotly_chart(fig_eq, use_container_width=True)

    with tab_anim:
        st.markdown("#### 🎬 Radar Animé : Trajectoire du Cours Live entre Stop-Loss, PRU et Take-Profit")
        if live_pos_rows:
            # Animation Plotly : fait glisser chaque action du PRU (0%) vers son Cours Live actuel sur l'échelle [SL -> TP]
            n_frames = 12
            radar_syms = [r["Symbole"] for r in live_pos_rows]
            base_sl = []
            base_tp = []
            base_pru = []
            base_live = []
            for r in live_pos_rows:
                pru_v = r["PRU ($)"]
                live_v = r["Cours Live ($)"]
                sl_v = r["Stop-Loss ($)"] if r["Stop-Loss ($)"] is not None else round(pru_v * 0.95, 2)
                tp_v = r["Take-Profit ($)"] if r["Take-Profit ($)"] is not None else round(pru_v * 1.10, 2)
                base_pru.append(pru_v)
                base_live.append(live_v)
                base_sl.append(sl_v)
                base_tp.append(tp_v)

            # Normalisation en % de progression entre SL (0%) et TP (100%) pour comparer toutes les actions sur une même piste animée
            def _norm_pos(val, sl_v, tp_v):
                span = max(tp_v - sl_v, 1e-6)
                return max(-15.0, min(115.0, (val - sl_v) / span * 100.0))

            pru_norm = [_norm_pos(base_pru[i], base_sl[i], base_tp[i]) for i in range(len(live_pos_rows))]
            live_norm = [_norm_pos(base_live[i], base_sl[i], base_tp[i]) for i in range(len(live_pos_rows))]

            frames_radar = []
            for step in range(n_frames + 1):
                alpha = step / n_frames
                curr_x = [pru_norm[i] + alpha * (live_norm[i] - pru_norm[i]) for i in range(len(live_pos_rows))]
                curr_px = [base_pru[i] + alpha * (base_live[i] - base_pru[i]) for i in range(len(live_pos_rows))]
                curr_pnl = [(curr_px[i] - base_pru[i]) * live_pos_rows[i]["Quantité"] for i in range(len(live_pos_rows))]
                colors = ["#00ff88" if curr_pnl[i] >= 0 else "#ff4b4b" for i in range(len(live_pos_rows))]
                frames_radar.append(go.Frame(
                    name=f"t{step}",
                    data=[
                        go.Bar(
                            y=radar_syms, x=curr_x, orientation="h",
                            marker_color=colors,
                            text=[f"Live: {curr_px[i]:.2f}$ | PnL: {curr_pnl[i]:+.1f}$" for i in range(len(live_pos_rows))],
                            textposition="auto",
                        )
                    ],
                ))

            fig_radar = go.Figure(
                data=[
                    go.Bar(
                        name="Cours Live (Progression SL → TP)",
                        y=radar_syms, x=live_norm, orientation="h",
                        marker_color=["#00ff88" if r["Gain/Perte ($)"] >= 0 else "#ff4b4b" for r in live_pos_rows],
                        text=[f"Live: {r['Cours Live ($)']:.2f}$ | PnL: {r['Gain/Perte ($)']:+.1f}$ ({r['Gain/Perte (%)']:+.1f}%)" for r in live_pos_rows],
                        textposition="auto",
                    ),
                    go.Scatter(
                        name="PRU (Prix d'Entrée)",
                        y=radar_syms, x=pru_norm, mode="markers",
                        marker=dict(symbol="diamond", size=13, color="#38bdf8", line=dict(color="white", width=1.5)),
                    ),
                ],
                frames=frames_radar,
            )
            fig_radar.add_vline(x=0, line_dash="dash", line_color="#ff4b4b", annotation_text="🛑 STOP-LOSS (0%)")
            fig_radar.add_vline(x=100, line_dash="dash", line_color="#00ff88", annotation_text="🎯 TAKE-PROFIT (100%)")
            fig_radar.update_layout(
                title="Animation : Déplacement du Cours (PRU → Live) entre Stop-Loss et Take-Profit",
                template="plotly_dark",
                height=max(300, 75 * len(live_pos_rows) + 140),
                xaxis=dict(title="Position sur le corridor Stop-Loss (0%) → Take-Profit (100%)", range=[-18, 122]),
                updatemenus=[{
                    "type": "buttons", "showactive": False, "x": 0.0, "y": 1.18, "xanchor": "left",
                    "buttons": [
                        {"label": "▶️ Animer PRU → Live", "method": "animate", "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": False, "transition": {"duration": 50}}]},
                        {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                    ],
                }],
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Ouvrez au moins une position (ou lancez un cycle automatique) pour visualiser le radar animé SL / PRU / TP.")

        st.markdown("#### 📈 Animation des Gains & Pertes Cumulés (Achats & Ventes)")
        if sell_records:
            df_sells = pd.DataFrame(sell_records).copy()
            df_sells["pnl"] = pd.to_numeric(df_sells.get("pnl", 0.0), errors="coerce").fillna(0.0)
            df_sells["cum_pnl"] = df_sells["pnl"].cumsum()
            df_sells["trade_idx"] = [f"#{i+1} {r.get('ticker', '')}" for i, (_, r) in enumerate(df_sells.iterrows())]

            frames_pnl = []
            for k in range(1, len(df_sells) + 1):
                sub = df_sells.iloc[:k]
                frames_pnl.append(go.Frame(
                    name=f"trade_{k}",
                    data=[
                        go.Bar(
                            x=sub["trade_idx"], y=sub["pnl"],
                            marker_color=["#00ff88" if v >= 0 else "#ff4b4b" for v in sub["pnl"]],
                            name="PnL par Vente ($)",
                        ),
                        go.Scatter(
                            x=sub["trade_idx"], y=sub["cum_pnl"],
                            mode="lines+markers", line=dict(color="#00d2ff", width=3),
                            name="Gain/Perte Cumulé ($)",
                        ),
                    ],
                ))

            fig_pnl_anim = go.Figure(
                data=[
                    go.Bar(
                        x=df_sells["trade_idx"], y=df_sells["pnl"],
                        marker_color=["#00ff88" if v >= 0 else "#ff4b4b" for v in df_sells["pnl"]],
                        name="PnL par Vente ($)",
                    ),
                    go.Scatter(
                        x=df_sells["trade_idx"], y=df_sells["cum_pnl"],
                        mode="lines+markers", line=dict(color="#00d2ff", width=3),
                        name="Gain/Perte Cumulé ($)",
                    ),
                ],
                frames=frames_pnl,
            )
            fig_pnl_anim.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_pnl_anim.update_layout(
                title="Replay Animé des Gains / Pertes Réalisés par Vente & Cumul Total",
                template="plotly_dark", height=380,
                yaxis_title="Résultat ($)",
                updatemenus=[{
                    "type": "buttons", "showactive": False, "x": 0.0, "y": 1.16, "xanchor": "left",
                    "buttons": [
                        {"label": "▶️ Rejouer les Gains/Pertes", "method": "animate", "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": False}]},
                        {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                    ],
                }],
            )
            st.plotly_chart(fig_pnl_anim, use_container_width=True)
        else:
            st.info("Aucune vente clôturée pour le moment : l'animation des gains/pertes réalisés s'affichera dès la première vente.")

        # Bilan consolidé par action (Achats, Ventes, PnL Réalisé, PnL Latent, Total Net)
        per_ticker = {}
        for h in pf_history:
            tk = h.get("ticker", "N/A")
            rec = per_ticker.setdefault(tk, {"Symbole": tk, "Achats": 0, "Ventes": 0, "PnL Réalisé ($)": 0.0, "PnL Latent ($)": 0.0})
            act = str(h.get("action", "")).upper()
            if "BUY" in act:
                rec["Achats"] += 1
            elif "SELL" in act:
                rec["Ventes"] += 1
                rec["PnL Réalisé ($)"] += float(h.get("pnl", 0.0) or 0.0)
        for r in live_pos_rows:
            tk = r["Symbole"]
            rec = per_ticker.setdefault(tk, {"Symbole": tk, "Achats": 0, "Ventes": 0, "PnL Réalisé ($)": 0.0, "PnL Latent ($)": 0.0})
            rec["PnL Latent ($)"] = r["Gain/Perte ($)"]
        if per_ticker:
            summary_rows = []
            for rec in per_ticker.values():
                rec["PnL Réalisé ($)"] = round(rec["PnL Réalisé ($)"], 2)
                rec["PnL Latent ($)"] = round(rec["PnL Latent ($)"], 2)
                rec["Résultat Total ($)"] = round(rec["PnL Réalisé ($)"] + rec["PnL Latent ($)"], 2)
                summary_rows.append(rec)
            st.markdown("#### 🧾 Bilan Gains / Pertes par Action (Achats + Ventes)")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with tab_rank:
        rankings = status.get("last_rankings", [])
        if rankings:
            st.dataframe(pd.DataFrame(rankings), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun scan effectué pour l'instant. Lancez le Mode Automatique ou cliquez sur 'Exécuter 1 Cycle'.")

    with tab_trades:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**⚡ Ordres de la Session Automatique**")
            events = status.get("trade_events", [])
            if events:
                df_ev = pd.DataFrame(events[::-1])
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Exporter les Trades Auto (CSV)",
                    data=df_ev.to_csv(index=False).encode("utf-8"),
                    file_name="autopilot_trades.csv",
                    mime="text/csv",
                    key="dl_auto_trades",
                )
            else:
                st.info("Aucun ordre exécuté par la session automatique en cours.")
        with col_t2:
            st.markdown("**📒 Historique Complet Portefeuille (Achats & Ventes)**")
            if pf_history:
                df_hist_all = pd.DataFrame(pf_history[::-1])
                st.dataframe(df_hist_all, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Exporter Historique Complet (CSV)",
                    data=df_hist_all.to_csv(index=False).encode("utf-8"),
                    file_name="portfolio_trades_history.csv",
                    mime="text/csv",
                    key="dl_pf_trades_all",
                )
            else:
                st.info("Aucun historique de transaction dans le portefeuille.")

    with tab_logs:
        logs = status.get("logs", [])
        if logs:
            df_logs = pd.DataFrame(logs[::-1])
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Exporter les Logs (CSV)",
                data=df_logs.to_csv(index=False).encode("utf-8"),
                file_name="autopilot_logs.csv",
                mime="text/csv",
                key="dl_auto_logs",
            )
        else:
            st.info("Journal vide.")

    # Auto-refresh optionnel pour suivre les cours et animations en temps réel
    auto_refresh = st.checkbox(
        "🔄 Rafraîchissement automatique en temps réel (toutes les 15s)",
        value=False,
        key="auto_refresh_dashboard",
    )
    if auto_refresh:
        time.sleep(15)
        st.rerun()


# --- IMPORTATEUR DE WATCHLISTS CSV (CONFORMITÉ & TRADUCTION AUTO) ---
_HAS_ST_DIALOG = hasattr(st, "dialog")


def _sync_custom_watchlists_state():
    """Synchronise les watchlists CSV importées en session dans MAJOR_STOCKS et PREDEFINED_PORTFOLIOS."""
    custom_stocks = st.session_state.get("custom_major_stocks", {})
    if custom_stocks:
        MAJOR_STOCKS.update(custom_stocks)
    custom_pfs = st.session_state.get("custom_portfolios", {})
    if custom_pfs:
        PREDEFINED_PORTFOLIOS.update(custom_pfs)


def _render_watchlist_import_body(target_context="sidebar"):
    """Contenu de la fenêtre de sélection, vérification et traduction des watchlists CSV."""
    st.caption(
        "Sélectionnez un ou plusieurs fichiers Watchlist (.csv) depuis votre répertoire. "
        "Chaque fichier est vérifié et traduit automatiquement au format de l'application (Yahoo Finance / WatchList)."
    )

    uploaded_files = st.file_uploader(
        "Sélectionner un ou plusieurs fichiers Watchlist (.csv)",
        type=["csv"],
        accept_multiple_files=True,
        key=f"csv_wl_uploader_{target_context}",
    )

    import_mode = st.radio(
        "Mode de chargement",
        [
            "🔄 Alternative (remplacer la sélection actuelle)",
            "➕ Complément (ajouter aux actions actuelles)",
        ],
        horizontal=True,
        key=f"csv_wl_mode_{target_context}",
    )

    save_to_disk = st.checkbox(
        "💾 Enregistrer la watchlist traduite dans le dossier WatchList/",
        value=True,
        key=f"csv_wl_save_{target_context}",
    )

    if not uploaded_files:
        st.info("Cliquez sur **Browse files** pour ouvrir l'explorateur et sélectionner vos fichiers CSV.")
        return

    valid_results = []
    for up_file in uploaded_files:
        res = parse_and_validate_watchlist_csv(up_file, up_file.name)
        if res["is_valid"]:
            valid_results.append((up_file.name, res))
            conv_msg = ""
            if res["conversions"]:
                sample_conv = ", ".join(f"`{r}` → `{c}`" for r, c in res["conversions"][:4])
                more = f" (+{len(res['conversions']) - 4})" if len(res["conversions"]) > 4 else ""
                conv_msg = f" | Traduction : {sample_conv}{more}"
            st.success(
                f"✅ **{up_file.name}** — **{len(res['stocks'])}** tickers valides "
                f"({res['format_detected']}){conv_msg}"
            )
            for w in res["warnings"]:
                st.warning(f"⚠️ {up_file.name} : {w}")
            with st.expander(f"Aperçu traduit : {up_file.name} ({len(res['entries'])} lignes)"):
                preview_df = pd.DataFrame([
                    {
                        "Nom": e["name"],
                        "Symbole brut": e["raw_symbol"],
                        "Symbole traduit (Yahoo)": e["symbol"],
                        "Bourse": e["exchange"],
                    }
                    for e in res["entries"]
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            err_txt = " ; ".join(res["errors"]) if res["errors"] else "Format non reconnu."
            st.error(f"❌ **{up_file.name}** — Fichier non conforme : {err_txt}")

    if not valid_results:
        st.warning("Aucun fichier conforme à charger. Vérifiez que votre CSV contient une colonne Symbol/Symbole/Ticker.")
        return

    if st.button(
        f"✅ Charger {len(valid_results)} watchlist(s) ({sum(len(r['stocks']) for _, r in valid_results)} tickers)",
        type="primary",
        use_container_width=True,
        key=f"csv_wl_apply_{target_context}",
    ):
        custom_stocks = st.session_state.setdefault("custom_major_stocks", {})
        custom_pfs = st.session_state.setdefault("custom_portfolios", {})

        all_imported_stocks = []
        saved_count = 0

        for _, res in valid_results:
            custom_stocks.update(res["major_stocks_map"])
            MAJOR_STOCKS.update(res["major_stocks_map"])
            custom_pfs[res["list_name"]] = list(res["stocks"])
            PREDEFINED_PORTFOLIOS[res["list_name"]] = list(res["stocks"])

            for s in res["stocks"]:
                if s not in all_imported_stocks:
                    all_imported_stocks.append(s)

            if save_to_disk:
                ok_save, _ = save_translated_watchlist_csv(res["translated_csv"], res["clean_filename"])
                if ok_save:
                    saved_count += 1

        if len(valid_results) == 1:
            target_pf_name = valid_results[0][1]["list_name"]
        else:
            target_pf_name = f"📂 Import combiné ({len(valid_results)} fichiers)"
            custom_pfs[target_pf_name] = list(all_imported_stocks)
            PREDEFINED_PORTFOLIOS[target_pf_name] = list(all_imported_stocks)

        if import_mode.startswith("🔄"):
            st.session_state[f"pending_pf_{target_context}"] = target_pf_name
            st.session_state[f"wl_complement_{target_context}"] = []
        else:
            existing_comp = list(st.session_state.get(f"wl_complement_{target_context}", []))
            for s in all_imported_stocks:
                if s not in existing_comp:
                    existing_comp.append(s)
            st.session_state[f"wl_complement_{target_context}"] = existing_comp

        st.session_state[f"wl_version_{target_context}"] = st.session_state.get(f"wl_version_{target_context}", 0) + 1
        st.session_state[f"wl_fallback_open_{target_context}"] = False

        save_note = f" ({saved_count} enregistrée(s) dans WatchList/)" if save_to_disk and saved_count else ""
        st.toast(f"✅ {len(all_imported_stocks)} ticker(s) chargé(s){save_note}", icon="📂")
        st.rerun()


if _HAS_ST_DIALOG:
    @st.dialog("📂 Importer des Watchlists CSV", width="large")
    def _watchlist_import_modal(target_context="sidebar"):
        _render_watchlist_import_body(target_context)
else:
    def _watchlist_import_modal(target_context="sidebar"):
        st.session_state[f"wl_fallback_open_{target_context}"] = True


# --- FONCTION PRINCIPALE ---
def main():
    if "portfolios_synced" not in st.session_state:
        load_portfolio()
        load_options_portfolio()
        st.session_state["portfolios_synced"] = True

    st.sidebar.title("🧭 Menu Principal")

    # Indicateur temps réel & Bouton STOP d'urgence du Mode Automatique visible partout
    _auto_engine = get_global_auto_trader()
    _auto_status = _auto_engine.get_status()
    if _auto_status["is_running"]:
        if _auto_status["is_paused"]:
            st.sidebar.warning(f"⏸️ Autopilot en pause (Cycle #{_auto_status['cycle_count']})")
        else:
            st.sidebar.success(f"🟢 Autopilot ACTIF (Cycle #{_auto_status['cycle_count']})")
        if st.sidebar.button("🛑 Interrompre Auto-Trading", use_container_width=True, key="sidebar_stop_autopilot"):
            _auto_engine.stop()
            st.rerun()

    menu = st.sidebar.radio("Sélectionnez un module :", [
        "📈 Terminal de Trading",
        "⚡ Mode Trading Automatique (Autopilot)",
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
    
    
    # --- NOUVEAU : Sélection rapide de portefeuilles + Import CSV ---
    _sync_custom_watchlists_state()
    pending_sidebar_pf = st.session_state.pop("pending_pf_sidebar", None)
    if pending_sidebar_pf and pending_sidebar_pf in PREDEFINED_PORTFOLIOS:
        st.session_state["pf_select_sidebar"] = pending_sidebar_pf

    col_pf_lbl, col_pf_btn = st.sidebar.columns([5, 1])
    with col_pf_lbl:
        st.markdown("**💡 Sélection rapide de portefeuille**")
    with col_pf_btn:
        if st.button("📂", key="btn_import_csv_sidebar", help="Importer des watchlists CSV depuis un répertoire"):
            if _HAS_ST_DIALOG:
                _watchlist_import_modal("sidebar")
            else:
                st.session_state["wl_fallback_open_sidebar"] = not st.session_state.get("wl_fallback_open_sidebar", False)

    if not _HAS_ST_DIALOG and st.session_state.get("wl_fallback_open_sidebar", False):
        with st.sidebar.expander("📂 Importer des Watchlists CSV", expanded=True):
            _render_watchlist_import_body("sidebar")

    portfolio_choice = st.sidebar.selectbox(
        "💡 Sélection rapide de portefeuille",
        options=list(PREDEFINED_PORTFOLIOS.keys()),
        index=0,
        key="pf_select_sidebar",
        label_visibility="collapsed"
    )

    last_sidebar_pf = st.session_state.get("last_pf_sidebar")
    if last_sidebar_pf is not None and last_sidebar_pf != portfolio_choice:
        st.session_state["wl_complement_sidebar"] = []
    st.session_state["last_pf_sidebar"] = portfolio_choice

    # Si un portefeuille est choisi, on utilise sa liste. Sinon par défaut Apple.
    default_selection = list(PREDEFINED_PORTFOLIOS[portfolio_choice]) if portfolio_choice != "Sélection Manuelle" else ["Apple Inc. (US)"]
    for extra_s in st.session_state.get("wl_complement_sidebar", []):
        if extra_s not in default_selection:
            default_selection.append(extra_s)

    # Filtrer les valeurs par défaut qui sont dans MAJOR_STOCKS pour le multiselect
    valid_defaults = [s for s in default_selection if s in MAJOR_STOCKS.keys()]
    sidebar_wl_ver = st.session_state.get("wl_version_sidebar", 0)

    # Remplacement par Multiselect réactif
    stock_choices = st.sidebar.multiselect(
        "Rechercher une ou plusieurs actions", 
        options=list(MAJOR_STOCKS.keys()), 
        default=valid_defaults,
        key=f"stocks_ms_{portfolio_choice}_{sidebar_wl_ver}"
    )

    # Sélection multiple directe de fichiers CSV de trackers (WatchList/)
    _avail_sidebar_csvs = list_watchlist_csv_files()
    selected_sidebar_csvs = st.sidebar.multiselect(
        "📂 Fichiers CSV de trackers (multi-sélection)",
        options=_avail_sidebar_csvs,
        default=[],
        key="sidebar_multi_csv_files",
        help="Sélectionnez plusieurs fichiers CSV du dossier WatchList/ pour fusionner tous leurs trackers.",
    )
    
    tickers = []
    
    # Ajout silencieux des tickers qui ne sont pas dans MAJOR_STOCKS (ex: Chine)
    for s in default_selection:
        if s not in MAJOR_STOCKS.keys():
            tickers.append(s)

    if selected_sidebar_csvs:
        _csv_bundle = load_tickers_from_csv_files(selected_sidebar_csvs)
        tickers.extend(_csv_bundle["symbols"])

    if "--- Saisir manuellement ---" in stock_choices:
        custom_ticker = st.sidebar.text_input("Ticker personnalisé (ex: EPA:MC)", "AAPL")
        tickers.append(convert_google_to_yahoo_ticker(custom_ticker))
        
    for choice in stock_choices:
        if choice != "--- Saisir manuellement ---":
            tickers.append(convert_google_to_yahoo_ticker(MAJOR_STOCKS[choice]))
            
    # Déduplication en préservant l'ordre
    tickers = list(dict.fromkeys(tickers))
    
    if menu == "📈 Terminal de Trading":
        st.title("🚀 XGBoost Stock Trader Pro")
        st.sidebar.header("Paramètres IA")
        
        period = st.sidebar.selectbox("Période d'historique", ["2y", "5y", "10y", "max"], index=3)
        interval = st.sidebar.selectbox("Intervalle", ["1d", "1wk"], index=0)
        initial_capital = st.sidebar.number_input("Capital Initial Total ($)", min_value=100, max_value=1000000, value=10000, step=100)
        optimize_model = st.sidebar.checkbox("🧠 Auto-Optimisation (Optuna Bayésien)")

        with st.sidebar.expander("⚙️ Stratégie & Gestion du Risque", expanded=False):
            lbl_choice = st.selectbox(
                "Cible ML (Labeling)",
                ["Triple Barrier (ATR 5j)", "Classique (J+1)"],
                index=0,
                help="Triple Barrier aligne la cible ML avec les stops ATR (+3 ATR vs -1.5 ATR sur 5j)."
            )
            calibrate_probs = st.checkbox("Calibration des probabilités", value=True, help="Calibre les probabilités de l'ensemble (Platt scaling).")
            use_trailing_stop = st.checkbox("Trailing Stop ATR (Chandelier)", value=True, help="Laisse courir les tendances avec un stop suiveur 2 ATR.")
            use_regime_filters = st.checkbox("Filtres SMA 200 & RSI en backtest", value=True, help="Bloque les achats sous SMA 200 ou si RSI >= 75.")
            use_vol_sizing = st.checkbox("Position Sizing Vol/ATR (2% risque)", value=False, help="Dimensionne chaque position selon le risque ATR et le VIX.")

        strategy_cfg = {
            "labeling_mode": "triple_barrier" if lbl_choice.startswith("Triple") else "next_day",
            "calibrate_probs": calibrate_probs,
            "use_trailing_stop": use_trailing_stop,
            "use_regime_filters": use_regime_filters,
            "use_vol_sizing": use_vol_sizing,
        }
        
        st.sidebar.divider()
        st.sidebar.header("Logique Walk-Forward (WFA)")
        use_wfa = st.sidebar.checkbox("Activer Walk-Forward Analysis (WFA)", value=True)
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
            run_single_mode(tickers[0], period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date, strategy_cfg=strategy_cfg)
        else:
            run_portfolio_mode(tickers, period, interval, initial_capital, optimize_model, use_wfa, wfa_train_window, wfa_step, wfa_start_date, wfa_end_date, strategy_cfg=strategy_cfg)
            
    elif menu == "⚡ Mode Trading Automatique (Autopilot)":
        page_auto_trading(sidebar_tickers=tickers, sidebar_csv_files=selected_sidebar_csvs)
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
