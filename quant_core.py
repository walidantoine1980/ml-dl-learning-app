"""
===============================================================================
MOTEUR QUANTITATIF CENTRAL (QUANT CORE ENGINE)
XGBoost Stock Predictor Pro - Clean Architecture
===============================================================================
Module pur Python (indépendant de Streamlit) regroupant :
- L'ingénierie des caractéristiques stationnaires (Feature Engineering)
- Le modèle d'Ensemble ML (XGBoost, LightGBM, Random Forest)
- Le moteur de Backtest Événementiel avec vrais Stops ATR dynamiques
- La Validation Walk-Forward (WFA) et l'optimisation bayésienne (Optuna)
- Le modèle d'allocation bayésien de Black-Litterman
- Le pricing analytique Black-Scholes et Greeks
- La persistance et la gestion du Paper Trading
===============================================================================
"""

import os
import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import scipy.stats as si
from scipy.optimize import brentq

# Modèles ML
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False


# =============================================================================
# 1. GESTION DES CHEMINS & PERSISTANCE DU PORTEFEUILLE
# =============================================================================

PORTFOLIO_FILE = "portfolio.json"
OPTIONS_PORTFOLIO_FILE = "options_portfolio.json"


def get_portfolio_path():
    return PORTFOLIO_FILE


def load_portfolio(filepath=PORTFOLIO_FILE):
    """Charge le portefeuille virtuel d'actions."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if not isinstance(data.get("positions"), dict):
                        data["positions"] = {}
                    if not isinstance(data.get("history"), list):
                        data["history"] = []
                    if "cash" not in data or not isinstance(data["cash"], (int, float)):
                        data["cash"] = 100000.0
                    return data
        except Exception as e:
            print(f"Erreur de chargement {filepath}: {e}")
    return {"cash": 100000.0, "positions": {}, "history": []}


def save_portfolio(portfolio, filepath=PORTFOLIO_FILE):
    """Sauvegarde atomique du portefeuille d'actions."""
    temp_file = f"{filepath}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=4)
    if os.path.exists(filepath):
        os.replace(temp_file, filepath)
    else:
        os.rename(temp_file, filepath)


def get_options_portfolio_path():
    return OPTIONS_PORTFOLIO_FILE


def load_options_portfolio(filepath=OPTIONS_PORTFOLIO_FILE):
    """Charge le portefeuille virtuel d'options."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if not isinstance(data.get("positions"), dict):
                        data["positions"] = {}
                    if not isinstance(data.get("history"), list):
                        data["history"] = []
                    if "cash" not in data or not isinstance(data["cash"], (int, float)):
                        data["cash"] = 100000.0
                    return data
        except Exception as e:
            print(f"Erreur de chargement {filepath}: {e}")
    return {"cash": 100000.0, "positions": {}, "history": []}


def save_options_portfolio(portfolio, filepath=OPTIONS_PORTFOLIO_FILE):
    """Sauvegarde atomique du portefeuille d'options."""
    temp_file = f"{filepath}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=4)
    if os.path.exists(filepath):
        os.replace(temp_file, filepath)
    else:
        os.rename(temp_file, filepath)


# =============================================================================
# 2. CONVERSION ET NORMALISATION DES TICKERS
# =============================================================================

def convert_google_to_yahoo_ticker(ticker):
    """Convertit un ticker Google Finance en format Yahoo Finance."""
    if ":" not in ticker:
        return ticker.strip().upper()
    exchange, symbol = ticker.split(":", 1)
    exchange = exchange.strip().upper()
    symbol = symbol.strip().upper()
    exchange_mapping = {
        'EPA': '.PA', 'LON': '.L', 'TSE': '.TO', 'FRA': '.F', 'BIT': '.MI',
        'BME': '.MC', 'AMS': '.AS', 'EBR': '.BR', 'LIS': '.LS', 'VTX': '.SW',
        'TYO': '.T', 'HKG': '.HK', 'ASX': '.AX', 'NASDAQ': '', 'NYSE': '', 'AMEX': ''
    }
    suffix = exchange_mapping.get(exchange, "")
    return f"{symbol}{suffix}"


def get_company_name_from_yahoo(yahoo_ticker):
    """Retrouve le nom lisible d'une entreprise via la table de tickers."""
    try:
        from tickers_db import MAJOR_STOCKS
        for name, google_ticker in MAJOR_STOCKS.items():
            if google_ticker != "CUSTOM":
                if convert_google_to_yahoo_ticker(google_ticker) == yahoo_ticker:
                    clean_name = name.split(" (")[0]
                    return f"{clean_name} ({yahoo_ticker})"
    except Exception:
        pass
    return yahoo_ticker


def get_option_multiplier_and_legislation(ticker):
    """
    Retourne le multiplicateur de contrat et la législation applicable.
    US: 100 actions/contrat (style américain).
    EU: 10 actions/contrat (style européen).
    """
    eu_suffixes = ['.PA', '.L', '.TO', '.F', '.MI', '.MC', '.AS', '.BR', '.LS', '.SW', '.DE']
    if any(ticker.endswith(suffix) for suffix in eu_suffixes):
        return 10, "Législation Européenne (10 actions/contrat - Exercice à l'échéance uniquement)"
    return 100, "Législation Américaine (100 actions/contrat - Exercice et revente libres)"


# =============================================================================
# 3. INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
# =============================================================================

def add_features(df):
    """
    Calcule les indicateurs techniques et ratios stationnaires.
    CORRECTIF D'INTÉGRITÉ MAJEUR :
    - Conserve INTÉGRALEMENT la dernière bougie t (Close d'aujourd'hui) sans la supprimer.
    - Ajoute le ratio NATR (Normalized ATR = ATR / Close) invariant d'échelle pour les arbres.
    - Conserve l'ATR brut pour le dimensionnement monétaire des Stops et Take-Profits.
    - Aplatit automatiquement les colonnes MultiIndex issues de yfinance.
    """
    df = df.copy()
    if hasattr(df.columns, "droplevel") and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.droplevel(1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # 1. Rendements journaliers
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Moyennes mobiles brutes (adaptatives pour supporter les historiques courts)
    w_20 = min(20, max(5, len(df) // 4))
    w_50 = min(50, max(10, len(df) // 2))
    w_200 = min(200, max(20, len(df) - 1))
    
    df['SMA_20'] = df['Close'].rolling(window=w_20, min_periods=w_20).mean().bfill()
    df['SMA_50'] = df['Close'].rolling(window=w_50, min_periods=w_50).mean().bfill()
    df['SMA_200'] = df['Close'].rolling(window=w_200, min_periods=min(50, w_200)).mean().bfill()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # 3. Ratios stationnaires (Invariants d'échelle pour Machine Learning)
    df['SMA_20_Ratio'] = (df['Close'] / (df['SMA_20'] + 1e-9)) - 1.0
    df['SMA_50_Ratio'] = (df['Close'] / (df['SMA_50'] + 1e-9)) - 1.0
    df['SMA_200_Ratio'] = (df['Close'] / (df['SMA_200'] + 1e-9)) - 1.0
    df['EMA_9_Ratio'] = (df['Close'] / (df['EMA_9'] + 1e-9)) - 1.0
    
    # 4. Volatilité historique
    df['Vol_20'] = df['Returns'].rolling(window=20).std()
    
    # 5. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. MACD normalisé par le cours
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp1 - exp2) / (df['Close'] + 1e-9)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 7. Bandes de Bollinger
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Percent'] = np.where(bb_range > 0, (df['Close'] - df['BB_Lower']) / (bb_range + 1e-9), 0.5)
    df['BB_Width'] = bb_range / (df['BB_Mid'] + 1e-9)
    
    # 8. True Range & ATR & NATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # NATR : Normalized ATR en ratio du cours (Stationnaire)
    df['NATR'] = df['ATR'] / (df['Close'] + 1e-9)
    
    # 9. ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].shift() - df['Low']
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm < 0, 0, minus_dm)
    plus_dm_actual = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm_actual = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr14 = true_range.rolling(14).sum() + 1e-9
    plus_di14 = 100 * (pd.Series(plus_dm_actual, index=df.index).rolling(14).sum() / tr14)
    minus_di14 = 100 * (pd.Series(minus_dm_actual, index=df.index).rolling(14).sum() / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-9)
    df['ADX'] = dx.rolling(14).mean()
    
    # 10. Volume Ratio
    vol_mean_20 = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (vol_mean_20 + 1e-9)
    
    # 11. Stochastique (%K)
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    stoch_range = highest_high - lowest_low
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (stoch_range + 1e-9))
    
    # 12. ROC (Rate of Change 10j)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # 13. VWAP Ratio
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vol_sum_14 = df['Volume'].rolling(window=14).sum() + 1e-9
    df['VWAP'] = (typical_price * df['Volume']).rolling(window=14).sum() / vol_sum_14
    df['VWAP_Ratio'] = (df['Close'] / (df['VWAP'] + 1e-9)) - 1.0
    
    # 14. Lags stationnaires de rendements
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        df[f'Lag_{i}_Return'] = df['Returns'].shift(i)
        
    # 15. Cible future (Target)
    future_diff = df['Close'].shift(-1) - df['Close']
    df['Target'] = np.where(future_diff > 0, 1.0, np.where(future_diff < 0, 0.0, np.nan))
    
    # IMPORTANT: On ne supprime que les NaN des FEATURES initiales (ex: 200 premiers jours de chauffe)
    # mais ON CONSERVE la dernière ligne t où Target est NaN !
    feature_cols = [c for c in df.columns if c != 'Target']
    df = df.dropna(subset=feature_cols).copy()
    
    return df


def get_macro_data(period="2y", interval="1d"):
    """
    Télécharge et prépare les variables macroéconomiques globales :
    - SPY (Marché US)
    - ^VIX (Indice de volatilité et de panique)
    - ^TNX (Taux 10 ans souverain américain)
    - DX-Y.NYB (Indice Dollar US transformé en variation stationnaire)
    """
    try:
        spy_raw = yf.download("SPY", period=period, interval=interval, progress=False)
        vix_raw = yf.download("^VIX", period=period, interval=interval, progress=False)
        tnx_raw = yf.download("^TNX", period=period, interval=interval, progress=False)
        dxy_raw = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
        
        def extract_close(raw_df):
            if raw_df is None or raw_df.empty:
                return pd.Series(dtype=float)
            if hasattr(raw_df.columns, "droplevel") and getattr(raw_df.columns, "nlevels", 1) > 1:
                raw_df = raw_df.copy()
                raw_df.columns = raw_df.columns.droplevel(1)
            if 'Close' in raw_df.columns:
                c = raw_df['Close']
                if isinstance(c, pd.DataFrame):
                    return c.iloc[:, 0]
                return c
            return pd.Series(dtype=float)
            
        spy = extract_close(spy_raw)
        vix = extract_close(vix_raw)
        tnx = extract_close(tnx_raw)
        dxy = extract_close(dxy_raw)
        
        if spy.empty:
            return None
            
        macro = pd.DataFrame(index=spy.index)
        macro['SPY_Return'] = spy.pct_change()
        macro['VIX'] = vix
        macro['TNX'] = tnx
        # Transformation du dollar en variation journalière stationnaire
        macro['DXY_Return'] = dxy.pct_change() if not dxy.empty else 0.0
        
        # Nettoyage timezone pour alignement multi-marchés (Europe/US/Asie)
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)
            
        return macro.ffill().fillna(0.0)
    except Exception as e:
        print(f"Erreur de téléchargement des données macro: {e}")
        return None


def join_macro_data(df, macro_df):
    """Jointure sécurisée des données macro avec alignement de calendrier et fuseau horaire."""
    if macro_df is None:
        return df
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if macro_df.index.tz is not None:
        macro_df = macro_df.copy()
        macro_df.index = macro_df.index.tz_localize(None)
        
    joined = df.join(macro_df, how='left')
    macro_cols = list(macro_df.columns)
    joined[macro_cols] = joined[macro_cols].ffill().fillna(0.0)
    return joined


# =============================================================================
# 4. DÉCLENCHEURS QUANTITATIFS MULTI-RÉGIMES (TRIGGERS ENGINE)
# =============================================================================

def evaluate_multi_regime_triggers(row, current_price, current_atr=None, trader_prob=None, current_vix=None, iv_annual=None):
    """
    Évalue l'ensemble des déclencheurs (triggers) institutionnels multi-régimes :
    1. Régime de Tendance Macro (SMA 200 / SMA 50)
    2. Bouclier RSI (Surchat > 75 & Survente Rebond < 30)
    3. Flux Institutionnel (Volume Surge >= 1.5x)
    4. Compression de Volatilité (Bollinger Squeeze)
    5. Coupe-Circuit Macro (VIX > 30)
    6. Régime de Volatilité Options (IV Rank / IV Level)
    7. Paliers de Sécurisation (Stop Loss, Breakeven Stop et Take Profit)
    """
    if current_atr is None or current_atr <= 0:
        current_atr = current_price * 0.02
        
    triggers = {}
    
    # 1. Régime de Tendance Macro
    sma_200 = float(row.get('SMA_200', 0.0))
    sma_50 = float(row.get('SMA_50', 0.0))
    if sma_200 > 0:
        if current_price > sma_200 and (sma_50 >= sma_200 or sma_50 == 0):
            triggers['trend_regime'] = {
                'status': 'BULLISH',
                'badge': '🟢 Tendance Saine (Bull Market)',
                'action': 'Achats autorisés avec pleine exposition.',
                'color': 'green'
            }
        elif current_price < sma_200 and (sma_50 <= sma_200 or sma_50 == 0):
            triggers['trend_regime'] = {
                'status': 'BEARISH',
                'badge': '🔴 Tendance Baissière (Bear Market)',
                'action': 'Prudence : Achats restreints ou taille réduite de 50%.',
                'color': 'red'
            }
        else:
            triggers['trend_regime'] = {
                'status': 'NEUTRAL',
                'badge': '🟡 Zone de Transition / Correction',
                'action': 'Consolidation en cours, confirmation requise.',
                'color': 'orange'
            }
    else:
        triggers['trend_regime'] = {
            'status': 'UNKNOWN',
            'badge': '⚪ Tendance Neutre',
            'action': 'Historique insuffisant pour SMA 200.',
            'color': 'gray'
        }
        
    # 2. Bouclier RSI
    rsi = float(row.get('RSI', 50.0))
    if rsi >= 75:
        triggers['rsi_shield'] = {
            'status': 'OVERBOUGHT_SHIELD',
            'badge': f'🛡️ Bouclier Surchat Actif (RSI {rsi:.1f} >= 75)',
            'action': 'Blocage des nouveaux achats au sommet. Prise de bénéfices recommandée.',
            'color': 'red'
        }
    elif rsi <= 30:
        triggers['rsi_shield'] = {
            'status': 'OVERSOLD_OPPORTUNITY',
            'badge': f'🎯 Opportunité Rebond (RSI {rsi:.1f} <= 30)',
            'action': "Survente extrême : favorable à un 'Buy the Dip' si le signal ML confirme.",
            'color': 'green'
        }
    else:
        triggers['rsi_shield'] = {
            'status': 'NORMAL',
            'badge': f'✅ RSI Équilibré ({rsi:.1f})',
            'action': 'Pas de surchauffe ni de panique.',
            'color': 'blue'
        }
        
    # 3. Flux Institutionnel (Volume Surge)
    vol_ratio = float(row.get('Volume_Ratio', 1.0))
    if vol_ratio >= 1.5:
        triggers['volume_surge'] = {
            'status': 'INSTITUTIONAL_INFLOW',
            'badge': f'🚀 Volume Institutionnel Détecté (+{((vol_ratio-1.0)*100):.0f}%)',
            'action': "Forte participation des 'Mains Fortes' (Smart Money).",
            'color': 'green'
        }
    else:
        triggers['volume_surge'] = {
            'status': 'NORMAL_VOLUME',
            'badge': f'📊 Volume Régulier ({vol_ratio:.2f}x)',
            'action': "Activité de liquidité standard.",
            'color': 'gray'
        }
        
    # 4. Compression de Volatilité (Bollinger Squeeze)
    bb_width = float(row.get('BB_Width', 0.10))
    bb_upper = float(row.get('BB_Upper', current_price * 1.05))
    bb_lower = float(row.get('BB_Lower', current_price * 0.95))
    if bb_width < 0.05:
        triggers['bollinger_squeeze'] = {
            'status': 'SQUEEZE_ACTIVE',
            'badge': f'⚡ Bollinger Squeeze Actif (Largeur {bb_width*100:.1f}%)',
            'action': 'Volatilité comprimée au maximum : explosion de cours imminente.',
            'color': 'purple'
        }
    elif current_price >= bb_upper:
        triggers['bollinger_squeeze'] = {
            'status': 'UPPER_BREAKOUT',
            'badge': '🔥 Breakout Haussier (Dépassement Bande Supérieure)',
            'action': 'Momentum de cassure haussière puissant.',
            'color': 'green'
        }
    elif current_price <= bb_lower:
        triggers['bollinger_squeeze'] = {
            'status': 'LOWER_BREAKOUT',
            'badge': '❄️ Cassure Baissière (Sous la Bande Inférieure)',
            'action': 'Pression vendeuse extrême.',
            'color': 'red'
        }
    else:
        triggers['bollinger_squeeze'] = {
            'status': 'NORMAL_BAND',
            'badge': f'↔️ Fluctuation Intra-Bandes ({bb_width*100:.1f}%)',
            'action': 'Évolution dans le canal de volatilité normal.',
            'color': 'gray'
        }
        
    # 5. Coupe-Circuit Macro (VIX)
    vix = float(current_vix if current_vix is not None else row.get('VIX', 18.0))
    if vix > 30:
        triggers['macro_vix'] = {
            'status': 'CIRCUIT_BREAKER_ON',
            'badge': f'🚨 COUPE-CIRCUIT VIX ACTIF ({vix:.1f} > 30)',
            'action': 'Liquidation préventive et passage forcé à 100% Cash.',
            'color': 'red'
        }
    elif vix > 20:
        triggers['macro_vix'] = {
            'status': 'ELEVATED_RISK',
            'badge': f'⚠️ Volatilité Macro Accrue (VIX {vix:.1f})',
            'action': 'Réduire les tailles de position de 25%.',
            'color': 'orange'
        }
    else:
        triggers['macro_vix'] = {
            'status': 'CALM_MARKET',
            'badge': f'🟢 Marché Directeur Serein (VIX {vix:.1f})',
            'action': 'Environnement macroéconomique favorable au portage.',
            'color': 'green'
        }
        
    # 6. Régime de Volatilité Options
    if iv_annual is not None and iv_annual > 0:
        if iv_annual >= 0.35:
            triggers['options_iv'] = {
                'status': 'HIGH_IV',
                'badge': f'💰 Volatilité Implicite Élevée ({iv_annual*100:.1f}%)',
                'action': 'Stratégies VENDEUSES de primes privilégiées (Credit Spreads, Iron Condors).',
                'color': 'green'
            }
        elif iv_annual <= 0.22:
            triggers['options_iv'] = {
                'status': 'LOW_IV',
                'badge': f'🎯 Volatilité Bon Marché ({iv_annual*100:.1f}%)',
                'action': 'Stratégies ACHETEUSES privilégiées (Long Calls / Puts / Straddles).',
                'color': 'blue'
            }
        else:
            triggers['options_iv'] = {
                'status': 'MODERATE_IV',
                'badge': f'⚖️ Volatilité Moyenne ({iv_annual*100:.1f}%)',
                'action': 'Spreads directionnels ou strangles adaptés.',
                'color': 'gray'
            }
            
    # 7. Niveaux de Protection Dynamique
    triggers['risk_levels'] = {
        'initial_sl': current_price - (1.5 * current_atr),
        'breakeven_trigger': current_price + (1.5 * current_atr),
        'initial_tp': current_price + (3.0 * current_atr),
        'atr': current_atr
    }
    
    return triggers


# =============================================================================
# 5. MOTEUR MACHINE LEARNING (ENSEMBLE & BACKTEST ÉVÉNEMENTIEL)
# =============================================================================

class MLTrader:
    """
    Moteur de Trading Quantitatif d'Ensemble.
    Combine XGBoost, LightGBM et Random Forest avec vote probabiliste souple (soft voting).
    Gère le backtest événementiel réaliste avec Stops ATR, Slippage et Commissions.
    """
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, 
            objective='binary:logistic', random_state=42
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        estimators = [('xgb', self.xgb_model), ('rf', self.rf_model)]
        if HAS_LIGHTGBM and lgb is not None:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=5, 
                random_state=42, verbose=-1
            )
            estimators.append(('lgb', self.lgb_model))
        else:
            self.lgb_model = None

        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_importances = None
        self.backtest_results = None
        self.advanced_metrics = {}
        self.closed_trades = []

    def get_feature_list(self, df):
        """Retourne la liste des features valides présentes dans le DataFrame."""
        base_features = [
            'Returns', 'SMA_20_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 'EMA_9_Ratio', 
            'Vol_20', 'RSI', 'MACD', 'Signal_Line', 'BB_Percent', 'BB_Width', 'NATR', 'ADX', 
            'Volume_Ratio', 'Stoch_K', 'ROC', 'VWAP_Ratio', 
            'Lag_1_Return', 'Lag_2_Return', 'Lag_3_Return'
        ]
        macro_features = ['SPY_Return', 'VIX', 'TNX', 'DXY_Return']
        return [f for f in base_features + macro_features if f in df.columns]

    def run_monte_carlo(self, returns, num_simulations=1000, ruin_threshold=0.8):
        """Évalue la probabilité de ruine (perte > 20% du capital) par Bootstrap Monte Carlo."""
        if len(returns) == 0:
            return 0.0
        n_days = len(returns)
        simulations = np.random.choice(returns, size=(num_simulations, n_days), replace=True)
        cum_simulations = np.cumprod(1 + simulations, axis=1)
        ruin_cases = np.any(cum_simulations < ruin_threshold, axis=1)
        return float(np.mean(ruin_cases))

    def calculate_advanced_metrics(self, test_data, closed_trades=None):
        """
        Calcule les métriques de performance et de risque institutionnelles.
        - Métriques basées sur les transactions réelles fermées (Win Rate, Kelly, Payoff)
        - Métriques de rendement journalier (Sharpe, Sortino, VaR, CVaR, Monte Carlo)
        """
        metrics = {}
        strat_returns = test_data['Strategy_Return'].fillna(0)
        
        # 1. Métriques par trade réel fermé (si disponibles)
        if closed_trades and len(closed_trades) > 0:
            trades_df = pd.DataFrame(closed_trades)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            trade_win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            wins = trades_df[trades_df['pnl'] > 0]['pnl_pct']
            losses = trades_df[trades_df['pnl'] < 0]['pnl_pct'].abs()
            avg_win = wins.mean() if not wins.empty else 0.0
            avg_loss = losses.mean() if not losses.empty else 1.0
            payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
            
            # Formule de Kelly classique basée sur les vrais paris fermés
            if payoff_ratio > 0 and trade_win_rate > 0:
                trade_kelly = trade_win_rate - ((1.0 - trade_win_rate) / payoff_ratio)
            else:
                trade_kelly = 0.0
                
            metrics['Total Trades'] = total_trades
            metrics['Trade Win Rate'] = trade_win_rate
            metrics['Payoff Ratio'] = payoff_ratio
            metrics['Trade Kelly %'] = max(0.0, trade_kelly)
            metrics['Half Kelly %'] = min(0.05, max(0.0, trade_kelly / 2.0))
            metrics['Kelly %'] = metrics['Trade Kelly %']
        else:
            # Fallback journalier
            winning_days = len(strat_returns[strat_returns > 0])
            total_trades_days = len(strat_returns[strat_returns != 0])
            trade_win_rate = winning_days / total_trades_days if total_trades_days > 0 else 0.0
            metrics['Total Trades'] = 0
            metrics['Trade Win Rate'] = trade_win_rate
            metrics['Trade Kelly %'] = 0.0
            metrics['Half Kelly %'] = 0.0
            metrics['Kelly %'] = 0.0

        metrics['Win Rate'] = trade_win_rate
        
        # 2. Drawdown Maximum
        cum_returns = (1 + strat_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        metrics['Max Drawdown'] = float(drawdown.min()) if not drawdown.empty else 0.0
        
        # 3. Sharpe & Sortino
        mean_ret = strat_returns.mean()
        std_ret = strat_returns.std()
        metrics['Sharpe Ratio'] = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        
        downside_returns = strat_returns[strat_returns < 0]
        downside_std = downside_returns.std()
        metrics['Sortino Ratio'] = float(mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        
        # 4. VaR 95% & CVaR (Expected Shortfall)
        if not strat_returns.empty:
            var_95 = float(np.percentile(strat_returns.dropna(), 5))
            metrics['VaR (95%)'] = var_95
            metrics['CVaR'] = float(strat_returns[strat_returns <= var_95].mean())
        else:
            metrics['VaR (95%)'] = 0.0
            metrics['CVaR'] = 0.0
            
        # 5. Probabilité de Ruine
        metrics['Probabilité de Ruine'] = self.run_monte_carlo(strat_returns.dropna().values)
        
        self.advanced_metrics = metrics
        return metrics

    def run_event_driven_backtest(
        self, test_data, initial_capital=100000.0, 
        use_atr_stops=True, atr_sl_mult=1.5, atr_tp_mult=3.0,
        commission_rate=0.0005, slippage_rate=0.0010,
        use_breakeven_stop=True
    ):
        """
        Moteur de Backtest Événementiel Réaliste :
        - Exécution à l'Open du signal calculé à la clôture de la veille (zéro biais d'anticipation).
        - Slippage (0.10%) et Commissions (0.05%) rigoureusement déduits à l'achat et à la vente.
        - Filtre d'Hystérésis (0.55 / 0.45) pour limiter le churning et les faux signaux.
        - Coupe-circuit Macro VIX > 30 (passage forcé à 100% Cash).
        - VRAIS STOPS DYNAMIQUES ATR exécutés en cours de séance :
            * Stop-Loss déclenché si Low <= Stop-Loss
            * Take-Profit déclenché si High >= Take-Profit
        """
        test_data = test_data.copy()
        cash = float(initial_capital)
        shares = 0.0
        
        portfolio_values = []
        daily_returns = []
        last_val = float(initial_capital)
        
        has_open = 'Open' in test_data.columns
        has_high_low = ('High' in test_data.columns) and ('Low' in test_data.columns)
        has_atr = 'ATR' in test_data.columns
        
        # 1. Hystérésis de Probabilité
        signals = []
        curr_sig = 0
        probs = test_data['Prob'].values if 'Prob' in test_data.columns else [0.5] * len(test_data)
        for p in probs:
            if p > 0.55:
                curr_sig = 1
            elif p < 0.45:
                curr_sig = 0
            signals.append(curr_sig)
        test_data['Signal'] = signals
        
        # Coupe-circuit Macro VIX
        if 'VIX' in test_data.columns:
            test_data.loc[test_data['VIX'] > 30, 'Signal'] = 0
            
        closed_trades = []
        entry_price = 0.0
        entry_date = None
        entry_index = None
        current_sl = 0.0
        current_tp = 0.0
        
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            date = test_data.index[i]
            
            # Latence : décision issue de la veille (i-1)
            raw_signal = 0 if i == 0 else test_data['Signal'].iloc[i-1]
            current_open = float(row['Open']) if has_open else float(row['Close'])
            current_close = float(row['Close'])
            current_high = float(row['High']) if has_high_low else current_close
            current_low = float(row['Low']) if has_high_low else current_close
            current_atr = float(row['ATR']) if has_atr and not pd.isna(row['ATR']) else (current_close * 0.02)
            
            # --- VÉRIFICATION DES POSITIONS OUVERTES ---
            if shares > 0:
                exit_price = None
                exit_reason = None
                
                # Mise à jour Breakeven Stop : dès que le High dépasse 1R (entry + 1.5 ATR), remonter le SL au prix d'entrée
                if use_atr_stops and use_breakeven_stop and current_high >= entry_price + (atr_sl_mult * current_atr):
                    current_sl = max(current_sl, entry_price)
                
                # A. Stop-Loss / Breakeven Stop ATR touché en séance
                if use_atr_stops and current_low <= current_sl:
                    # Exécution au stop loss (ou à l'ouverture si gap baissier)
                    exit_price = min(current_open, current_sl) * (1.0 - slippage_rate)
                    exit_reason = "Breakeven Stop (0 Risque)" if current_sl >= entry_price else f"Stop Loss ({atr_sl_mult:.1f} ATR)"
                # B. Take-Profit ATR touché en séance
                elif use_atr_stops and current_high >= current_tp:
                    # Exécution au take profit (ou à l'ouverture si gap haussier)
                    exit_price = max(current_open, current_tp) * (1.0 - slippage_rate)
                    exit_reason = f"Take Profit ({atr_tp_mult:.1f} ATR)"
                # C. Coupe-circuit Macro VIX (> 30)
                elif 'VIX' in row and float(row['VIX']) > 30:
                    exit_price = current_open * (1.0 - slippage_rate)
                    exit_reason = "VIX > 30 (Circuit Breaker)"
                # D. Signal ML inversé (Probabilité < 0.45)
                elif raw_signal == 0:
                    exit_price = current_open * (1.0 - slippage_rate)
                    exit_reason = "Signal Exit (< 0.45)"
                    
                # Exécution de la sortie
                if exit_price is not None:
                    proceeds = shares * exit_price
                    commission = proceeds * commission_rate
                    cash += proceeds - commission
                    
                    pnl = (exit_price - entry_price) * shares - (2 * commission)
                    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    diff = date - entry_date if entry_date is not None else None
                    duration = diff.days if hasattr(diff, 'days') else max(1, i - (entry_index if entry_index is not None else i - 1))
                    
                    closed_trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'reason': exit_reason,
                        'duration_days': duration
                    })
                    shares = 0.0
                    entry_price = 0.0
                    entry_date = None
                    entry_index = None
                    current_sl = 0.0
                    current_tp = 0.0

            # --- VÉRIFICATION DES NOUVEAUX ACHATS ---
            if raw_signal == 1 and shares == 0:
                exec_price = current_open * (1.0 + slippage_rate)
                investable = cash / (1.0 + commission_rate)
                shares = investable / exec_price
                commission = investable * commission_rate
                cash -= (shares * exec_price + commission)
                
                entry_price = exec_price
                entry_date = date
                entry_index = i
                # Calcul des niveaux de Stop-Loss et Take-Profit basés sur l'ATR
                current_sl = entry_price - (atr_sl_mult * current_atr)
                current_tp = entry_price + (atr_tp_mult * current_atr)

            # Évaluation Mark-to-Market de fin de séance
            current_val = cash + (shares * current_close)
            portfolio_values.append(current_val)
            ret = (current_val - last_val) / last_val if last_val > 0 else 0.0
            daily_returns.append(ret)
            last_val = current_val

        test_data['Event_Portfolio_Value'] = portfolio_values
        test_data['Strategy_Return'] = daily_returns
        test_data['Cum_Strategy_Return'] = test_data['Event_Portfolio_Value'] / initial_capital
        first_close = float(test_data['Close'].iloc[0]) if len(test_data) > 0 and float(test_data['Close'].iloc[0]) > 0 else 1.0
        test_data['Cum_Market_Return'] = test_data['Close'] / first_close
        
        self.closed_trades = closed_trades
        self.calculate_advanced_metrics(test_data, closed_trades)
        return test_data

    def train(
        self, data, optimize=False, use_wfa=False, 
        wfa_train_window="5Y", wfa_step="6M", 
        wfa_start_date=None, wfa_end_date=None,
        progress_callback=None
    ):
        """
        Entraîne le modèle avec ou sans Walk-Forward Analysis (WFA) et Optuna.
        - Exclut proprement les NaN de la cible pour l'entraînement.
        - Conserve l'accès au dernier bar pour la prédiction en direct.
        """
        features = self.get_feature_list(data)
        self.features = list(features)
        
        # Filtre d'entraînement propre : seules les lignes ayant une Target connue sont entraînées
        valid_train = data.dropna(subset=['Target'])
        X = valid_train[features]
        y = valid_train['Target'].astype(int)
        
        if not use_wfa:
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Robustesse : si la cible ne contient qu'une seule classe (marché purement haussier ou baissier)
            if len(np.unique(y_train)) < 2:
                y_train = y_train.copy()
                y_train.iloc[0] = 1 - int(y_train.iloc[0])
            
            if optimize:
                if HAS_OPTUNA and optuna is not None:
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
                    tscv = TimeSeriesSplit(n_splits=3)

                    def objective(trial):
                        param = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                            'max_depth': trial.suggest_int('max_depth', 3, 7)
                        }
                        m = xgb.XGBClassifier(**param, objective='binary:logistic', random_state=42)
                        return cross_val_score(m, X_train, y_train, cv=tscv, scoring='accuracy').mean()

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=10)
                    best_params = study.best_params
                    self.xgb_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', random_state=42)

                estimators = [('xgb', self.xgb_model), ('rf', self.rf_model)]
                if HAS_LIGHTGBM and self.lgb_model is not None:
                    estimators.append(('lgb', self.lgb_model))
                self.model = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
                self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)
                
            # Importances des variables
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': importances}
            ).sort_values(by='Importance', ascending=True)
            
            # Backtest sur le jeu de test
            test_data = valid_train.iloc[split:].copy()
            clean_test = test_data[features]
            test_data['Prob'] = self.model.predict_proba(clean_test)[:, 1]
            test_data = self.run_event_driven_backtest(test_data)
            
            self.backtest_results = test_data
            self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
            self.is_trained = True
            return features
            
        else:
            # --- WALK FORWARD ANALYSIS (WFA) DYNAMIQUE ---
            window_map = {"2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5), "7Y": pd.DateOffset(years=7)}
            step_map = {"1M": pd.DateOffset(months=1), "2M": pd.DateOffset(months=2), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}
            
            train_window = window_map.get(wfa_train_window, pd.DateOffset(years=5))
            step_size = step_map.get(wfa_step, pd.DateOffset(months=6))
            
            last_ts = pd.Timestamp(valid_train.index[-1])
            first_ts = pd.Timestamp(valid_train.index[0])
            
            if wfa_end_date is None:
                wfa_end_date = last_ts.date()
            if wfa_start_date is None:
                computed_start = (pd.Timestamp(wfa_end_date) - train_window).date()
                min_start = (first_ts + train_window).date()
                wfa_start_date = max(min_start, computed_start)
                if wfa_start_date >= wfa_end_date:
                    wfa_start_date = (first_ts + pd.DateOffset(months=6)).date()
                
            current_date = pd.to_datetime(wfa_start_date)
            if current_date.tzinfo is None and data.index.tzinfo is not None:
                current_date = current_date.tz_localize(data.index.tzinfo)
            end_date = pd.to_datetime(wfa_end_date)
            if end_date.tzinfo is None and data.index.tzinfo is not None:
                end_date = end_date.tz_localize(data.index.tzinfo)
                
            all_test_data = []
            all_importances = []
            y_test_all = []
            y_pred_all = []
            
            total_days = max(1, (end_date - current_date).days)
            start_days = total_days
            
            while current_date < end_date:
                train_start = current_date - train_window
                test_end = current_date + step_size
                
                mask_train = (valid_train.index >= train_start) & (valid_train.index < current_date)
                mask_test = (valid_train.index >= current_date) & (valid_train.index < test_end)
                
                X_tr, y_tr = X[mask_train], y[mask_train]
                X_te, y_te_slice = X[mask_test], y[mask_test]
                
                if len(X_tr) < 50 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                    current_date += step_size
                    continue
                    
                self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, objective='binary:logistic', random_state=42)
                self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                estimators = [('xgb', self.xgb_model), ('rf', self.rf_model)]
                if HAS_LIGHTGBM and lgb is not None:
                    self.lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
                    estimators.append(('lgb', self.lgb_model))
                
                m = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
                m.fit(X_tr, y_tr)
                
                prob = m.predict_proba(X_te)[:, 1]
                pred = m.predict(X_te)
                
                test_slice = valid_train[mask_test].copy()
                test_slice['Prob'] = prob
                all_test_data.append(test_slice)
                
                y_test_all.extend(y_te_slice.values)
                y_pred_all.extend(pred)
                
                imp = np.mean([est.feature_importances_ for est in m.estimators_], axis=0)
                all_importances.append(imp)
                
                current_date += step_size
                if progress_callback and start_days > 0:
                    prog = min(1.0, 1.0 - ((end_date - current_date).days / start_days))
                    progress_callback(prog)
                    
            if len(all_test_data) == 0:
                return features
                
            final_test_data = pd.concat(all_test_data)
            self.model = m # Modèle final le plus récent
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': np.mean(all_importances, axis=0)}
            ).sort_values(by='Importance', ascending=True)
            
            final_test_data = self.run_event_driven_backtest(final_test_data)
            self.backtest_results = final_test_data
            self.accuracy = accuracy_score(y_test_all, y_pred_all) if len(y_test_all) > 0 else 0.0
            self.is_trained = True
            return features

    def predict(self, last_row, features):
        """
        Prédit la probabilité haussière pour la bougie suivante.
        Prend la bougie t courante et applique le coupe-circuit VIX si > 30.
        Garantit l'alignement exact des colonnes attendues par le modèle.
        """
        if not self.is_trained:
            return None
        features_expected = getattr(self, 'features', features)
        # Réalignement défensif garantissant que toutes les variables sont présentes
        X_input = last_row.reindex(columns=features_expected, fill_value=0.0)
        prob = float(self.model.predict_proba(X_input)[0][1])
        
        # Coupe-circuit Macro VIX > 30 en temps réel
        if 'VIX' in last_row.columns:
            vix_val = float(last_row['VIX'].iloc[0]) if isinstance(last_row['VIX'], pd.Series) else float(last_row['VIX'])
            if vix_val > 30:
                return 0.0
        return prob

    def save(self, filepath):
        """Sauvegarde l'état du modèle et ses résultats."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        state = {
            'model': self.model,
            'accuracy': self.accuracy,
            'feature_importances': self.feature_importances,
            'backtest_results': self.backtest_results,
            'is_trained': self.is_trained,
            'advanced_metrics': getattr(self, 'advanced_metrics', {}),
            'closed_trades': getattr(self, 'closed_trades', [])
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load(filepath):
        """Charge un modèle persisté sur le disque."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        trader = MLTrader()
        trader.model = state['model']
        trader.accuracy = state['accuracy']
        trader.feature_importances = state['feature_importances']
        trader.backtest_results = state['backtest_results']
        trader.is_trained = state['is_trained']
        trader.advanced_metrics = state.get('advanced_metrics', {})
        trader.closed_trades = state.get('closed_trades', [])
        return trader


# =============================================================================
# 5. ALLOCATION BAYÉSIENNE DE BLACK-LITTERMAN
# =============================================================================

def get_black_litterman_weights(returns_df, market_weights, views_dict, tau=0.05, risk_aversion=2.5):
    """
    Calcule les poids de portefeuille optimaux par le modèle de Black-Litterman.
    returns_df : rendements historiques des actions sélectionnées
    market_weights : prior d'équilibre (ex: Parité des Risques 1/Vol)
    views_dict : vues subjectives calibrées issues de l'IA {ticker: excess_return}
    """
    try:
        tickers = list(returns_df.columns)
        N = len(tickers)
        if N == 0:
            return {}
        if N == 1:
            return {tickers[0]: 1.0}
            
        Sigma = returns_df.cov().values * 252
        if np.isnan(Sigma).any():
            return market_weights
        # Régularisation diagonale minimale pour garantir la stricte inversibilité
        Sigma += np.eye(N) * 1e-6
        
        W_mkt = np.array([market_weights.get(t, 1.0/N) for t in tickers])
        Pi = risk_aversion * Sigma.dot(W_mkt)
        
        P = np.eye(N)
        Q = np.array([views_dict.get(t, 0.0) for t in tickers])
        
        if np.all(Q == 0.0):
            return market_weights
            
        Omega = np.diag(np.diag(tau * P.dot(Sigma).dot(P.T)))
        
        tau_Sigma_inv = np.linalg.pinv(tau * Sigma)
        Omega_inv = np.linalg.pinv(Omega)
        
        left_term = np.linalg.pinv(tau_Sigma_inv + P.T.dot(Omega_inv).dot(P))
        right_term = tau_Sigma_inv.dot(Pi) + P.T.dot(Omega_inv).dot(Q)
        
        posterior_expected_returns = left_term.dot(right_term)
        W_bl = np.linalg.pinv(risk_aversion * Sigma).dot(posterior_expected_returns)
        
        # Contrainte Long-Only (pas de short)
        W_bl = np.clip(W_bl, 0.0, None)
        total_w = np.sum(W_bl)
        if total_w > 0:
            W_bl = W_bl / total_w
        else:
            W_bl = W_mkt
            
        return {tickers[i]: float(W_bl[i]) for i in range(N)}
    except Exception as e:
        print(f"Erreur d'optimisation Black-Litterman: {e}")
        return market_weights


# =============================================================================
# 6. PRICING ANALYTIQUE D'OPTIONS (BLACK-SCHOLES & GREEKS)
# =============================================================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Pricing analytique d'options européennes selon Black-Scholes avec Greeks complets.
    Greeks calculés : Delta, Gamma, Theta (quotidien / 365), Vega.
    """
    T = max(float(T), 1e-5)
    sigma = max(float(sigma), 1e-4)
    S = max(float(S), 1e-4)
    K = max(float(K), 1e-4)
    r = float(r)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        delta = si.norm.cdf(d1, 0.0, 1.0)
        theta = (- (S * sigma * si.norm.pdf(d1, 0.0, 1.0)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) / 365.0
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        delta = -si.norm.cdf(-d1, 0.0, 1.0)
        theta = (- (S * sigma * si.norm.pdf(d1, 0.0, 1.0)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)) / 365.0
        
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    
    return max(0.0, float(price)), float(delta), float(gamma), float(theta), float(vega)


def find_strike_for_delta(S, T, r, sigma, option_type, target_delta):
    """
    Résout par la méthode de Brent le Strike exact K correspondant au Delta théorique cible.
    target_delta : positif pour Call (ex: 0.15), négatif pour Put (ex: -0.15).
    """
    def obj(K):
        _, d, _, _, _ = black_scholes(S, K, T, r, sigma, option_type)
        return d - target_delta
    try:
        K_opt = brentq(obj, S * 0.05, S * 4.0)
        return float(K_opt)
    except Exception:
        return float(S)


def run_options_backtest_simulation(
    closes, strategy="Credit Spread", strat_type="PUT", 
    strat_action="Vendre (Short)", target_delta_abs=0.15, 
    target_dte=45, take_profit_pct=50, stop_loss_pct=200, 
    capital=100000.0, multiplier=100, bid_ask_spread=0.02, 
    commission_per_leg=1.0, progress_callback=None
):
    """
    Simulateur de stratégies d'options systématiques avec gestion réaliste des frictions.
    """
    returns = closes.pct_change()
    hist_vol = (returns.rolling(window=30).std() * np.sqrt(252)).bfill()
    r = 0.05
    
    cash = float(capital)
    equity_curve = []
    trades = []
    current_position = None
    total_days = len(closes)
    
    for i, (date, spot) in enumerate(closes.items()):
        if progress_callback and i % 25 == 0:
            progress_callback(i / total_days)
            
        sigma = max(float(hist_vol.iloc[i]), 0.05)
        
        if current_position is None:
            legs = []
            if strategy == "Single Leg":
                td = target_delta_abs if strat_type == "CALL" else -target_delta_abs
                K = find_strike_for_delta(spot, target_dte/365.0, r, sigma, strat_type.lower(), td)
                action = 'short' if 'Vendre' in strat_action else 'long'
                legs.append({'type': strat_type.lower(), 'action': action, 'strike': K})
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
                
            total_cost_in = 0.0
            total_comm = len(legs) * commission_per_leg
            for leg in legs:
                p, _, _, _, _ = black_scholes(spot, leg['strike'], target_dte/365.0, r, sigma, leg['type'])
                # Application du spread bid-ask
                adj_p = p * (1.0 - bid_ask_spread) if leg['action'] == 'short' else p * (1.0 + bid_ask_spread)
                leg['premium_in'] = adj_p
                leg_cost = adj_p * multiplier
                if leg['action'] == 'short':
                    total_cost_in += leg_cost
                else:
                    total_cost_in -= leg_cost
                    
            total_cost_in -= total_comm
            current_position = {
                'entry_date': date,
                'entry_spot': spot,
                'legs': legs,
                'net_cost_in': total_cost_in,
                'days_passed': 0
            }
            cash += total_cost_in
            equity_curve.append(cash)
        else:
            current_position['days_passed'] += 1
            days_remaining = target_dte - current_position['days_passed']
            
            if days_remaining <= 0:
                closing_cashflow = 0.0
                total_comm = len(current_position['legs']) * commission_per_leg
                for leg in current_position['legs']:
                    intrinsic = max(0.0, spot - leg['strike']) if leg['type'] == 'call' else max(0.0, leg['strike'] - spot)
                    val = intrinsic * multiplier
                    closing_cashflow += (-val if leg['action'] == 'short' else val)
                closing_cashflow -= total_comm
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
                closing_cashflow = 0.0
                for leg in current_position['legs']:
                    p, _, _, _, _ = black_scholes(spot, leg['strike'], days_remaining/365.0, r, sigma, leg['type'])
                    adj_p = p * (1.0 + bid_ask_spread) if leg['action'] == 'short' else p * (1.0 - bid_ask_spread)
                    val = adj_p * multiplier
                    closing_cashflow += (-val if leg['action'] == 'short' else val)
                    
                current_pnl = current_position['net_cost_in'] + closing_cashflow
                close_position = False
                reason = ""
                
                if current_position['net_cost_in'] > 0:
                    max_profit = current_position['net_cost_in']
                    if current_pnl >= max_profit * (take_profit_pct / 100.0):
                        close_position = True
                        reason = "Take Profit"
                    elif current_pnl <= -max_profit * (stop_loss_pct / 100.0):
                        close_position = True
                        reason = "Stop Loss"
                else:
                    max_loss = abs(current_position['net_cost_in'])
                    if max_loss > 0:
                        if current_pnl >= max_loss * (take_profit_pct / 100.0):
                            close_position = True
                            reason = "Take Profit"
                        elif current_pnl <= -max_loss * (stop_loss_pct / 100.0):
                            close_position = True
                            reason = "Stop Loss"
                            
                if close_position:
                    comm = len(current_position['legs']) * commission_per_leg
                    closing_cashflow -= comm
                    cash += closing_cashflow
                    trades.append({
                        'entry_date': current_position['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': date.strftime('%Y-%m-%d'),
                        'strategy': strategy,
                        'reason': reason,
                        'pnl': current_pnl - comm
                    })
                    current_position = None
                    equity_curve.append(cash)
                else:
                    equity_curve.append(cash + closing_cashflow)
                    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_equity': equity_curve[-1] if equity_curve else capital
    }


# =============================================================================
# 7. EXÉCUTION DE TRANSACTIONS LOCALES (PAPER TRADING)
# =============================================================================

def execute_equity_trade(portfolio, ticker, qty, action, current_price, current_atr=None, slippage=0.0010, commission=0.0005):
    """
    Exécute un ordre d'achat ou de vente dans le portefeuille virtuel avec traçabilité complète.
    Intègre les niveaux de Stop-Loss et Take-Profit calculés sur l'ATR.
    """
    pf = portfolio
    action = action.upper()
    exec_price = current_price * (1.0 + slippage if "BUY" in action else 1.0 - slippage)
    cost = qty * exec_price
    comm_cost = cost * commission
    
    if "BUY" in action:
        total_cost = cost + comm_cost
        if pf.get('cash', 0.0) >= total_cost:
            pf['cash'] -= total_cost
            atr_val = current_atr if current_atr and current_atr > 0 else (exec_price * 0.02)
            sl_price = exec_price - (1.5 * atr_val)
            tp_price = exec_price + (3.0 * atr_val)
            
            if ticker in pf['positions']:
                old_qty = pf['positions'][ticker]['qty']
                old_price = pf['positions'][ticker]['avg_price']
                new_qty = old_qty + qty
                new_price = ((old_qty * old_price) + total_cost) / new_qty
                pf['positions'][ticker] = {
                    'qty': new_qty, 
                    'avg_price': new_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price
                }
            else:
                pf['positions'][ticker] = {
                    'qty': qty, 
                    'avg_price': exec_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price
                }
                
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY',
                'ticker': ticker,
                'qty': qty,
                'price': exec_price,
                'total': total_cost,
                'stop_loss': sl_price,
                'take_profit': tp_price
            })
            return True, f"Achat virtuel de {qty} actions {ticker} à {exec_price:.2f} $ validé (SL: {sl_price:.2f} $, TP: {tp_price:.2f} $)."
        else:
            return False, "Liquidités virtuelles insuffisantes."
    else:
        # Vente / Liquidation
        if ticker in pf.get('positions', {}) and pf['positions'][ticker].get('qty', 0) > 0:
            pos_qty = pf['positions'][ticker]['qty']
            qty_to_sell = min(qty, pos_qty)
            revenue = qty_to_sell * exec_price
            net_revenue = revenue - comm_cost
            pf['cash'] += net_revenue
            
            pnl = net_revenue - (qty_to_sell * pf['positions'][ticker]['avg_price'])
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': action,
                'ticker': ticker,
                'qty': qty_to_sell,
                'price': exec_price,
                'total': net_revenue,
                'pnl': pnl
            })
            if qty_to_sell >= pos_qty:
                del pf['positions'][ticker]
            else:
                pf['positions'][ticker]['qty'] -= qty_to_sell
            return True, f"Vente virtuelle de {qty_to_sell} actions {ticker} exécutée (PnL: {pnl:+.2f} $)."
        else:
            return False, "Aucune position ouverte sur ce titre."


def execute_option_trade(portfolio, ticker, opt_type, strike, days_to_expiry, qty, premium, underlying_price, action="BUY", multiplier=None):
    """
    Exécute un ordre d'achat ou de vente de contrat d'options dans le portefeuille d'options.
    Garantit une structure saine (positions est un dictionnaire et non une liste).
    """
    pf = portfolio
    if not isinstance(pf.get('positions'), dict):
        pf['positions'] = {}
    if not isinstance(pf.get('history'), list):
        pf['history'] = []
    if 'cash' not in pf or not isinstance(pf['cash'], (int, float)):
        pf['cash'] = 100000.0

    if multiplier is None:
        multiplier, _ = get_option_multiplier_and_legislation(ticker)

    opt_type = opt_type.lower()
    cost_per_contract = premium * multiplier
    total_cost = qty * cost_per_contract

    if action.upper() == "BUY":
        if pf['cash'] >= total_cost:
            pf['cash'] -= total_cost
            contract_id = f"{ticker}_{opt_type.upper()}_{strike}_{days_to_expiry}d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            pf['positions'][contract_id] = {
                'ticker': ticker,
                'type': opt_type,
                'strike': float(strike),
                'days_to_expiry': int(days_to_expiry),
                'premium': float(premium),
                'qty': int(qty),
                'multiplier': int(multiplier),
                'underlying_price_at_buy': float(underlying_price),
                'buy_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'BUY OPTION',
                'ticker': ticker,
                'contract': f"{opt_type.upper()} Strike {strike}",
                'qty': int(qty),
                'premium_paid': float(premium),
                'total_cost': float(total_cost)
            })
            return True, contract_id, f"{qty} contrat(s) {opt_type.upper()} {ticker} acheté(s) avec succès."
        else:
            return False, None, "Liquidités virtuelles insuffisantes pour cet achat d'option."
    elif action.upper() in ["SELL", "CLOSE"]:
        contract_id = None
        for cid, data in pf['positions'].items():
            if data.get('ticker') == ticker and data.get('type') == opt_type and abs(data.get('strike', 0) - strike) < 1e-4:
                contract_id = cid
                break
        if contract_id and contract_id in pf['positions']:
            pos = pf['positions'][contract_id]
            pos_qty = pos.get('qty', 0)
            qty_to_sell = min(qty, pos_qty)
            revenue = qty_to_sell * premium * multiplier
            pf['cash'] += revenue
            buy_val = qty_to_sell * pos.get('premium', premium) * multiplier
            pnl = revenue - buy_val
            pf['history'].append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'action': 'SELL OPTION',
                'ticker': ticker,
                'contract': f"{opt_type.upper()} Strike {strike}",
                'qty': int(qty_to_sell),
                'premium_sold': float(premium),
                'total_received': float(revenue),
                'pnl': float(pnl)
            })
            if qty_to_sell >= pos_qty:
                del pf['positions'][contract_id]
            else:
                pf['positions'][contract_id]['qty'] -= qty_to_sell
            return True, contract_id, f"{qty_to_sell} contrat(s) {opt_type.upper()} {ticker} revendu(s) avec succès (PnL: {pnl:+.2f} $)."
        else:
            return False, None, "Contrat d'option introuvable dans le portefeuille."
    return False, None, f"Action {action} non supportée."

