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
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import scipy.stats as si
from scipy.optimize import brentq

# Modèles ML
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
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


import time

# =============================================================================
# 1. GESTION DES CHEMINS & PERSISTANCE DU PORTEFEUILLE (MULTI-EMPLACEMENTS)
# =============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = "portfolio.json"
OPTIONS_PORTFOLIO_FILE = "options_portfolio.json"


def _get_persistent_paths(filename):
    """
    Retourne la liste ordonnée des emplacements de sauvegarde persistants
    pour garantir la conservation des investissements après relance ou mise à jour
    (dossier utilisateur ~/.xgboost_trader, dossier workspace parent, volume data/, et racine du projet).
    """
    paths = []
    # 1. Variable d'environnement explicite (si définie sur serveur/cloud)
    env_dir = os.environ.get("XGBOOST_PORTFOLIO_DIR", "").strip()
    if env_dir:
        paths.append(os.path.join(os.path.abspath(env_dir), filename))

    # 2. Dossier persistant dans le profil utilisateur (survit à toute mise à jour ou changement de dossier)
    try:
        user_home = os.path.expanduser("~")
        if user_home and user_home != "~":
            paths.append(os.path.join(user_home, ".xgboost_trader", filename))
    except Exception:
        pass

    # 3. Dossier persistant dans le répertoire parent du projet (survit aux nouvelles releases sur Google Drive / disque)
    try:
        parent_dir = os.path.dirname(PROJECT_DIR)
        if parent_dir and parent_dir != PROJECT_DIR and os.path.dirname(parent_dir) != parent_dir:
            paths.append(os.path.join(parent_dir, ".xgboost_trader_data", filename))
    except Exception:
        pass

    # 4. Volume de données local / Docker (/app/data ou <project>/data)
    paths.append(os.path.join(PROJECT_DIR, "data", filename))

    # 5. Racine du projet (chemin absolu indépendant du répertoire courant CWD)
    paths.append(os.path.join(PROJECT_DIR, filename))

    # 6. Répertoire de travail courant (si différent de PROJECT_DIR)
    cwd_path = os.path.abspath(filename)
    if cwd_path not in [os.path.abspath(p) for p in paths]:
        paths.append(cwd_path)

    # Déduplication tout en préservant l'ordre
    unique_paths = []
    seen = set()
    for p in paths:
        norm_p = os.path.normcase(os.path.abspath(p))
        if norm_p not in seen:
            seen.add(norm_p)
            unique_paths.append(os.path.abspath(p))
    return unique_paths


def _is_default_portfolio_path(filepath, default_filename):
    """Vérifie si le chemin demandé correspond au portefeuille principal (et non à un fichier de test isolé)."""
    if filepath is None or filepath == default_filename:
        return True
    abs_fp = os.path.normcase(os.path.abspath(filepath))
    for p in _get_persistent_paths(default_filename):
        if abs_fp == os.path.normcase(os.path.abspath(p)):
            return True
    return False


def _normalize_portfolio_dict(data):
    """Garantit l'intégrité structurelle d'un dictionnaire de portefeuille."""
    if not isinstance(data, dict):
        return {"cash": 100000.0, "positions": {}, "history": []}
    normalized = dict(data)
    if not isinstance(normalized.get("positions"), dict):
        normalized["positions"] = {}
    if not isinstance(normalized.get("history"), list):
        normalized["history"] = []
    if "cash" not in normalized or not isinstance(normalized["cash"], (int, float)) or np.isnan(normalized["cash"]):
        normalized["cash"] = 100000.0
    else:
        normalized["cash"] = float(normalized["cash"])
    return normalized


def _is_empty_reset_state(data):
    """Détermine si un portefeuille correspond à un état vierge (100 000 $, 0 position, 0 historique)."""
    if not isinstance(data, dict):
        return True
    positions = data.get("positions")
    history = data.get("history")
    cash = data.get("cash", 100000.0)
    has_pos = isinstance(positions, dict) and len(positions) > 0
    has_hist = isinstance(history, list) and len(history) > 0
    cash_changed = isinstance(cash, (int, float)) and abs(float(cash) - 100000.0) > 0.01
    return not (has_pos or has_hist or cash_changed)


def _read_single_portfolio_file(filepath):
    """Lit et valide un fichier JSON de portefeuille s'il existe."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return _normalize_portfolio_dict(raw)
    except Exception as e:
        print(f"Erreur de chargement {filepath}: {e}")
    return None


def _write_single_portfolio_file(portfolio, filepath):
    """Écriture atomique sécurisée d'un fichier JSON de portefeuille."""
    try:
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_file = f"{filepath}.{os.getpid()}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=4)
        os.replace(temp_file, filepath)
        return True
    except Exception:
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
        return False


def _load_best_persistent_portfolio(default_filename):
    """
    Charge la version la plus récente et la plus complète du portefeuille
    parmi tous les emplacements persistants, et synchronise automatiquement
    tous les emplacements pour survivre aux relances et mises à jour.
    """
    candidates = []
    paths = _get_persistent_paths(default_filename)
    for idx, p in enumerate(paths):
        data = _read_single_portfolio_file(p)
        if data is not None:
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0.0
            updated_at = float(data.get("_updated_at", 0.0)) if isinstance(data.get("_updated_at"), (int, float)) else 0.0
            is_empty = _is_empty_reset_state(data)
            pos_count = len(data.get("positions", {}))
            hist_count = len(data.get("history", []))
            
            # Priorité :
            # Tier 2 : Fichier horodaté par une action utilisateur (_updated_at > 0) ET non vierge (ou réinitialisé explicitement)
            # Tier 1 : Fichier existant contenant des positions ou de l'historique (migration sans perte)
            # Tier 0 : Fichier vierge réinitialisé explicitement
            # Tier -1 : Fichier template vierge non initialisé (ex: écrasé par une archive de mise à jour)
            if updated_at > 0 and not is_empty:
                tier = 3
            elif not is_empty:
                tier = 2
            elif updated_at > 0 and data.get("_last_action") == "RESET":
                tier = 1
            else:
                tier = -1

            # Si un fichier réinitialisé explicitement (tier 1) est plus récent que tous les fichiers avec _updated_at (tier 3)
            # on utilise directement updated_at pour départager les actions horodatées
            if updated_at > 0:
                score = (1, updated_at, pos_count, hist_count, mtime, -idx)
            elif not is_empty:
                score = (0, mtime, pos_count, hist_count, mtime, -idx)
            else:
                score = (-1, 0.0, 0, 0, mtime, -idx)

            candidates.append((score, data, p))

    if not candidates:
        return {"cash": 100000.0, "positions": {}, "history": []}

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_data = candidates[0][1]

    # Synchroniser tous les emplacements persistants avec le meilleur état trouvé
    if not _is_empty_reset_state(best_data) or best_data.get("_updated_at"):
        if "_updated_at" not in best_data:
            best_data["_updated_at"] = time.time()
        for p in paths:
            _write_single_portfolio_file(best_data, p)

    return best_data


def _save_persistent_portfolio(portfolio, filepath, default_filename, allow_reset=False):
    """
    Sauvegarde atomique du portefeuille.
    Protection stricte : refuse d'écraser un portefeuille contenant des investissements
    par un compte vierge à 100 000 $ sauf si allow_reset=True (bouton de réinitialisation).
    """
    pf = _normalize_portfolio_dict(portfolio)

    if not _is_default_portfolio_path(filepath, default_filename):
        if not allow_reset and _is_empty_reset_state(pf):
            existing = _read_single_portfolio_file(filepath)
            if existing is not None and not _is_empty_reset_state(existing):
                return existing
        _write_single_portfolio_file(pf, filepath)
        return pf

    if not allow_reset and _is_empty_reset_state(pf):
        existing_best = _load_best_persistent_portfolio(default_filename)
        if not _is_empty_reset_state(existing_best):
            return existing_best

    pf["_updated_at"] = time.time()
    if allow_reset and _is_empty_reset_state(pf):
        pf["_last_action"] = "RESET"
    else:
        pf["_last_action"] = "UPDATE"

    for p in _get_persistent_paths(default_filename):
        _write_single_portfolio_file(pf, p)
    return pf


def get_portfolio_path():
    return os.path.join(PROJECT_DIR, PORTFOLIO_FILE)


def load_portfolio(filepath=PORTFOLIO_FILE):
    """Charge le portefeuille virtuel d'actions avec persistance multi-emplacements."""
    if not _is_default_portfolio_path(filepath, PORTFOLIO_FILE):
        data = _read_single_portfolio_file(filepath)
        return data if data is not None else {"cash": 100000.0, "positions": {}, "history": []}
    return _load_best_persistent_portfolio(PORTFOLIO_FILE)


def save_portfolio(portfolio, filepath=PORTFOLIO_FILE, allow_reset=False):
    """Sauvegarde atomique du portefeuille d'actions sur tous les emplacements persistants."""
    return _save_persistent_portfolio(portfolio, filepath, PORTFOLIO_FILE, allow_reset=allow_reset)


def reset_portfolio(filepath=PORTFOLIO_FILE):
    """Réinitialise explicitement le compte Paper Trading (Virtuel) à 100 000 $."""
    return save_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=filepath, allow_reset=True)


def get_options_portfolio_path():
    return os.path.join(PROJECT_DIR, OPTIONS_PORTFOLIO_FILE)


def load_options_portfolio(filepath=OPTIONS_PORTFOLIO_FILE):
    """Charge le portefeuille virtuel d'options avec persistance multi-emplacements."""
    if not _is_default_portfolio_path(filepath, OPTIONS_PORTFOLIO_FILE):
        data = _read_single_portfolio_file(filepath)
        return data if data is not None else {"cash": 100000.0, "positions": {}, "history": []}
    return _load_best_persistent_portfolio(OPTIONS_PORTFOLIO_FILE)


def save_options_portfolio(portfolio, filepath=OPTIONS_PORTFOLIO_FILE, allow_reset=False):
    """Sauvegarde atomique du portefeuille d'options sur tous les emplacements persistants."""
    return _save_persistent_portfolio(portfolio, filepath, OPTIONS_PORTFOLIO_FILE, allow_reset=allow_reset)


def reset_options_portfolio(filepath=OPTIONS_PORTFOLIO_FILE):
    """Réinitialise explicitement le compte Paper Trading (Options) à 100 000 $."""
    return save_options_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=filepath, allow_reset=True)


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

def _infer_periods_per_year(index):
    """Déduit automatiquement la fréquence d'annualisation (252 journalier vs 52 hebdomadaire)."""
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
            diffs = index.to_series().diff().dt.days.dropna()
            if not diffs.empty and float(diffs.median()) >= 5.0:
                return 52.0
    except Exception:
        pass
    return 252.0


def _compute_triple_barrier_labels(close_arr, high_arr, low_arr, atr_arr, horizon=5, tp_mult=3.0, sl_mult=1.5):
    """
    Triple Barrier Labeling (López de Prado) :
    - Barrière haute : entry + tp_mult * ATR  -> label 1.0
    - Barrière basse : entry - sl_mult * ATR  -> label 0.0
    - Barrière verticale : horizon séances    -> 1.0 si Close_fin > entry sinon 0.0
    - La dernière bougie t reste strictement NaN (pas de fuite future).
    """
    n = len(close_arr)
    labels = np.full(n, np.nan, dtype=float)
    for i in range(n - 1):
        entry = close_arr[i]
        atr_i = atr_arr[i]
        if np.isnan(entry) or entry <= 0:
            continue
        if np.isnan(atr_i) or atr_i <= 0:
            atr_i = entry * 0.02
        upper = entry + tp_mult * atr_i
        lower = entry - sl_mult * atr_i
        end_j = min(i + horizon, n - 1)
        hit = np.nan
        for j in range(i + 1, end_j + 1):
            low_hit = low_arr[j] <= lower
            high_hit = high_arr[j] >= upper
            if low_hit and high_hit:
                hit = 0.0  # Conservateur en cas d'ambiguïté intra-bougie
                break
            elif low_hit:
                hit = 0.0
                break
            elif high_hit:
                hit = 1.0
                break
        if np.isnan(hit):
            hit = 1.0 if close_arr[end_j] > entry else 0.0
        labels[i] = hit
    return labels


def add_features(df, labeling_mode="next_day", tb_horizon=5, tb_tp_mult=3.0, tb_sl_mult=1.5):
    """
    Calcule les indicateurs techniques et ratios stationnaires (strictement causaux, sans bfill).
    - Conserve INTÉGRALEMENT la dernière bougie t (Close d'aujourd'hui) sans la supprimer.
    - Lissage exponentiel de Wilder pour RSI (14), ATR (14) et ADX (14).
    - Enrichi : volatilité Garman-Klass, ratios de régimes de vol, microstructure OHLC (CLV, gaps, mèches),
      flux de volume (CMF, MFI, OBV Z-score), momentum multi-horizons (5j/21j/63j), distance 52s, calendrier.
    - Supporte le labeling classique J+1 ('next_day') et Triple Barrier ('triple_barrier').
    """
    df = df.copy()
    if hasattr(df.columns, "droplevel") and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.droplevel(1)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 1. Rendements journaliers
    df['Returns'] = df['Close'].pct_change()

    # 2. Moyennes mobiles brutes (strictement causales, sans bfill)
    w_20 = min(20, max(5, len(df) // 4))
    w_50 = min(50, max(10, len(df) // 2))
    w_200 = min(200, max(20, len(df) - 1))

    df['SMA_20'] = df['Close'].rolling(window=w_20, min_periods=w_20).mean()
    df['SMA_50'] = df['Close'].rolling(window=w_50, min_periods=min(20, w_50)).mean()
    df['SMA_200'] = df['Close'].rolling(window=w_200, min_periods=min(50, w_200)).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

    # 3. Ratios stationnaires & Pentes normalisées
    df['SMA_20_Ratio'] = (df['Close'] / (df['SMA_20'] + 1e-9)) - 1.0
    df['SMA_50_Ratio'] = (df['Close'] / (df['SMA_50'] + 1e-9)) - 1.0
    df['SMA_200_Ratio'] = (df['Close'] / (df['SMA_200'] + 1e-9)) - 1.0
    df['EMA_9_Ratio'] = (df['Close'] / (df['EMA_9'] + 1e-9)) - 1.0
    df['SMA_50_Slope'] = df['SMA_50'].pct_change(5)
    df['SMA_200_Slope'] = df['SMA_200'].pct_change(10)

    # 4. Volatilité historique, Ratios de régimes & Garman-Klass (OHLC)
    df['Vol_5'] = df['Returns'].rolling(window=5, min_periods=5).std()
    df['Vol_20'] = df['Returns'].rolling(window=20, min_periods=20).std()
    w_vol60 = min(60, max(20, len(df) // 4))
    df['Vol_60'] = df['Returns'].rolling(window=w_vol60, min_periods=min(20, w_vol60)).std()
    df['Vol_Ratio_5_20'] = df['Vol_5'] / (df['Vol_20'] + 1e-9)
    df['Vol_Ratio_20_60'] = df['Vol_20'] / (df['Vol_60'] + 1e-9)

    hl_log = np.log((df['High'].clip(lower=1e-9)) / (df['Low'].clip(lower=1e-9)))
    co_log = np.log((df['Close'].clip(lower=1e-9)) / (df['Open'].clip(lower=1e-9)))
    gk_var = 0.5 * (hl_log ** 2) - (2.0 * np.log(2.0) - 1.0) * (co_log ** 2)
    df['GK_Vol_20'] = np.sqrt(gk_var.rolling(window=20, min_periods=10).mean().clip(lower=0.0))

    # 5. RSI (14) — Lissage exponentiel de Wilder & dynamique de momentum
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI_Change_5'] = df['RSI'].diff(5)

    # 6. MACD normalisé par le cours + Histogramme & Accélération
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp1 - exp2) / (df['Close'] + 1e-9)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['MACD_Hist_Slope'] = df['MACD_Hist'].diff(3)

    # 7. Bandes de Bollinger & Z-Score
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    bb_range = df['BB_Upper'] - df['BB_Lower']
    df['BB_Percent'] = np.where(bb_range > 0, (df['Close'] - df['BB_Lower']) / (bb_range + 1e-9), 0.5)
    df['BB_Width'] = bb_range / (df['BB_Mid'] + 1e-9)
    df['Price_ZScore_20'] = (df['Close'] - df['BB_Mid']) / (df['BB_Std'] + 1e-9)

    # 8. True Range & ATR (Lissage de Wilder) & NATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    df['NATR'] = df['ATR'] / (df['Close'] + 1e-9)

    # 9. ADX (Average Directional Index — Lissage de Wilder)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].shift() - df['Low']
    plus_dm = np.where(plus_dm < 0, 0.0, plus_dm)
    minus_dm = np.where(minus_dm < 0, 0.0, minus_dm)
    plus_dm_actual = pd.Series(np.where(plus_dm > minus_dm, plus_dm, 0.0), index=df.index)
    minus_dm_actual = pd.Series(np.where(minus_dm > plus_dm, minus_dm, 0.0), index=df.index)

    tr14 = true_range.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() + 1e-9
    plus_di14 = 100.0 * (plus_dm_actual.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / tr14)
    minus_di14 = 100.0 * (minus_dm_actual.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / tr14)
    dx = 100.0 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-9)
    df['ADX'] = dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    df['DI_Diff'] = plus_di14 - minus_di14  # Direction signée de tendance (+DI - -DI)

    # 10. Volume Ratio, CMF (20), MFI (14) & OBV Z-score
    vol_mean_20 = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (vol_mean_20 + 1e-9)

    hl_span = (df['High'] - df['Low']).replace(0, np.nan)
    df['CLV'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_span).fillna(0.0).clip(-1.0, 1.0)
    mf_vol = df['CLV'] * df['Volume']
    df['CMF_20'] = mf_vol.rolling(window=20, min_periods=10).sum() / (df['Volume'].rolling(window=20, min_periods=10).sum() + 1e-9)

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    raw_mf = typical_price * df['Volume']
    tp_diff = typical_price.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(window=14, min_periods=14).sum()
    neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(window=14, min_periods=14).sum()
    df['MFI_14'] = 100.0 - (100.0 / (1.0 + pos_mf / (neg_mf + 1e-9)))

    obv = (np.sign(df['Close'].diff().fillna(0.0)) * df['Volume']).cumsum()
    obv_mean = obv.rolling(window=20, min_periods=10).mean()
    obv_std = obv.rolling(window=20, min_periods=10).std()
    df['OBV_ZScore'] = ((obv - obv_mean) / (obv_std + 1e-9)).clip(-5.0, 5.0)

    # 11. Microstructure OHLC (Mèches & Gap d'ouverture)
    df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['Close'] + 1e-9)
    df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['Close'] + 1e-9)
    df['Gap_Open'] = (df['Open'] / (df['Close'].shift(1) + 1e-9)) - 1.0

    # 12. Stochastique (%K) & Momentum multi-horizons
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    stoch_range = highest_high - lowest_low
    df['Stoch_K'] = 100.0 * ((df['Close'] - lowest_low) / (stoch_range + 1e-9))

    df['ROC'] = df['Close'].pct_change(periods=10) * 100.0
    df['Ret_5d'] = df['Close'].pct_change(periods=5)
    df['Ret_21d'] = df['Close'].pct_change(periods=21)
    w_63 = min(63, max(15, len(df) // 4))
    df['Ret_63d'] = df['Close'].pct_change(periods=w_63)
    w_52w = min(252, max(20, len(df) // 2))
    high_52w = df['High'].rolling(window=w_52w, min_periods=20).max()
    low_52w = df['Low'].rolling(window=w_52w, min_periods=20).min()
    df['Dist_High_52w'] = (df['Close'] / (high_52w + 1e-9)) - 1.0
    df['Pos_52w'] = ((df['Close'] - low_52w) / (high_52w - low_52w + 1e-9)).clip(0.0, 1.0)

    # 12b. Régime directionnel & Breakout (Kaufman Efficiency Ratio, Donchian 20j, Downside Vol)
    abs_change_10 = np.abs(df['Close'] - df['Close'].shift(10))
    sum_abs_diff_10 = np.abs(df['Close'].diff()).rolling(window=10, min_periods=10).sum()
    df['ER_10'] = (abs_change_10 / (sum_abs_diff_10 + 1e-9)).clip(0.0, 1.0)

    donch_high_20 = df['High'].rolling(window=20, min_periods=10).max()
    donch_low_20 = df['Low'].rolling(window=20, min_periods=10).min()
    df['Donchian_Pos_20'] = ((df['Close'] - donch_low_20) / (donch_high_20 - donch_low_20 + 1e-9)).clip(0.0, 1.0)

    neg_ret_sq = np.minimum(df['Returns'].fillna(0.0), 0.0) ** 2
    df['Downside_Vol_20'] = np.sqrt(neg_ret_sq.rolling(window=20, min_periods=10).mean())

    # 13. VWAP Ratio
    vol_sum_14 = df['Volume'].rolling(window=14).sum() + 1e-9
    df['VWAP'] = (typical_price * df['Volume']).rolling(window=14).sum() / vol_sum_14
    df['VWAP_Ratio'] = (df['Close'] / (df['VWAP'] + 1e-9)) - 1.0

    # 14. Lags stationnaires de rendements
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        df[f'Lag_{i}_Return'] = df['Returns'].shift(i)

    # 15. Saisonnalité / Calendrier (encodage cyclique)
    if isinstance(df.index, pd.DatetimeIndex):
        dow = df.index.dayofweek
        df['DayOfWeek_Sin'] = np.sin(2.0 * np.pi * dow / 5.0)
        df['DayOfWeek_Cos'] = np.cos(2.0 * np.pi * dow / 5.0)
        df['TurnOfMonth'] = np.where((df.index.day <= 3) | (df.index.day >= 28), 1.0, 0.0)
    else:
        df['DayOfWeek_Sin'] = 0.0
        df['DayOfWeek_Cos'] = 1.0
        df['TurnOfMonth'] = 0.0

    # 16. Cible future (Target) : Classique J+1 ou Triple Barrier ATR
    if labeling_mode == "triple_barrier":
        df['Target'] = _compute_triple_barrier_labels(
            df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values,
            horizon=tb_horizon, tp_mult=tb_tp_mult, sl_mult=tb_sl_mult
        )
    else:
        future_diff = df['Close'].shift(-1) - df['Close']
        df['Target'] = np.where(future_diff > 0, 1.0, np.where(future_diff < 0, 0.0, np.nan))

    # IMPORTANT: On ne supprime que les NaN des FEATURES initiales (période de chauffe)
    # mais ON CONSERVE la dernière ligne t où Target est NaN !
    feature_cols = [c for c in df.columns if c != 'Target']
    df = df.dropna(subset=feature_cols).copy()

    return df


def get_macro_data(period="2y", interval="1d"):
    """
    Télécharge et prépare les variables macroéconomiques globales :
    - SPY (Marché US + rendements 1j/5j/20j pour force relative)
    - ^VIX & ^VIX3M (Volatilité et structure à terme contango/backwardation)
    - ^TNX & ^IRX (Taux 10 ans US et pente de courbe des taux 10Y-3M)
    - DX-Y.NYB (Indice Dollar US en variation stationnaire)
    """
    try:
        spy_raw = yf.download("SPY", period=period, interval=interval, progress=False)
        vix_raw = yf.download("^VIX", period=period, interval=interval, progress=False)
        tnx_raw = yf.download("^TNX", period=period, interval=interval, progress=False)
        dxy_raw = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
        try:
            vix3m_raw = yf.download("^VIX3M", period=period, interval=interval, progress=False)
        except Exception:
            vix3m_raw = None
        try:
            irx_raw = yf.download("^IRX", period=period, interval=interval, progress=False)
        except Exception:
            irx_raw = None

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
        vix3m = extract_close(vix3m_raw)
        irx = extract_close(irx_raw)

        if spy.empty:
            return None

        macro = pd.DataFrame(index=spy.index)
        macro['SPY_Close'] = spy
        spy_sma200 = spy.rolling(window=min(200, max(20, len(spy) // 2)), min_periods=20).mean()
        macro['SPY_Above_SMA200'] = np.where(spy >= spy_sma200, 1.0, 0.0)
        macro['SPY_Return'] = spy.pct_change()
        macro['SPY_Ret_5d'] = spy.pct_change(5)
        macro['SPY_Ret_20d'] = spy.pct_change(20)
        macro['VIX'] = vix
        macro['TNX'] = tnx
        macro['DXY_Return'] = dxy.pct_change() if not dxy.empty else 0.0

        # Structure à terme du VIX (^VIX / ^VIX3M) : > 1.0 = backwardation (stress)
        if not vix.empty and not vix3m.empty:
            vix3m_aligned = vix3m.reindex(macro.index).ffill()
            macro['VIX_Term_Ratio'] = (macro['VIX'] / (vix3m_aligned + 1e-9)).clip(0.5, 2.0)
        else:
            macro['VIX_Term_Ratio'] = 1.0

        # Pente de la courbe des taux (^TNX - ^IRX)
        if not tnx.empty and not irx.empty:
            irx_aligned = irx.reindex(macro.index).ffill()
            macro['Yield_Spread'] = macro['TNX'] - irx_aligned
        else:
            macro['Yield_Spread'] = 0.0

        # Nettoyage timezone pour alignement multi-marchés (Europe/US/Asie)
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)

        return macro.ffill().fillna(0.0)
    except Exception as e:
        print(f"Erreur de téléchargement des données macro: {e}")
        return None


def apply_alpha_vs_spy_target(df, horizon=5):
    """
    Cible 'Alpha pur vs SPY' (Surperformance relative sur `horizon` séances) :
    - Target = 1.0 si rendement_futur_action(t -> t+h) > rendement_futur_SPY(t -> t+h)
    - Target = 0.0 sinon
    - Les `horizon` dernières bougies restent strictement NaN (zéro fuite future) et sont conservées.
    """
    df = df.copy()
    h = max(1, int(horizon))
    stock_fut = (df['Close'].shift(-h) / (df['Close'] + 1e-9)) - 1.0
    if 'SPY_Close' in df.columns and (df['SPY_Close'] > 0).any():
        spy_fut = (df['SPY_Close'].shift(-h) / (df['SPY_Close'] + 1e-9)) - 1.0
        excess = stock_fut - spy_fut
    else:
        excess = stock_fut
    df['Target'] = np.where(excess.isna(), np.nan, np.where(excess > 0.0, 1.0, 0.0))
    return df


def join_macro_data(df, macro_df, labeling_mode=None, alpha_horizon=5):
    """Jointure sécurisée des données macro + calcul de la Force Relative, du Beta glissant et cible Alpha vs SPY optionnelle."""
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

    # Force relative & Beta glissant vs SPY (strictement causaux)
    if 'Returns' in joined.columns and 'SPY_Return' in joined.columns:
        joined['RS_vs_SPY_1d'] = joined['Returns'] - joined['SPY_Return']
        w_beta = min(60, max(20, len(joined) // 4))
        cov_spy = joined['Returns'].rolling(window=w_beta, min_periods=15).cov(joined['SPY_Return'])
        var_spy = joined['SPY_Return'].rolling(window=w_beta, min_periods=15).var()
        joined['Rolling_Beta_60'] = (cov_spy / (var_spy + 1e-9)).clip(-3.0, 5.0).fillna(1.0)
    if 'Ret_5d' in joined.columns and 'SPY_Ret_5d' in joined.columns:
        joined['RS_vs_SPY_5d'] = joined['Ret_5d'] - joined['SPY_Ret_5d']
    if 'Ret_21d' in joined.columns and 'SPY_Ret_20d' in joined.columns:
        joined['RS_vs_SPY_20d'] = joined['Ret_21d'] - joined['SPY_Ret_20d']
    if 'VIX' in joined.columns:
        joined['VIX_Change_5d'] = joined['VIX'].pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 2.0)

    if labeling_mode == "alpha_spy":
        joined = apply_alpha_vs_spy_target(joined, horizon=alpha_horizon)

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
    Combine XGBoost, LightGBM et Random Forest avec vote probabiliste souple (soft voting),
    calibration optionnelle, élagage colinéaire, Optuna régularisé (neg_log_loss) et
    backtest événementiel strictement causal (Stops ATR, Trailing Chandelier, Vol Sizing, Slippage, Commissions).
    """
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, min_child_weight=5,
            gamma=0.1, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0,
            objective='binary:logistic', random_state=42
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=120, max_depth=5, min_samples_leaf=10, max_features='sqrt', random_state=42
        )
        estimators = [('xgb', self.xgb_model), ('rf', self.rf_model)]
        if HAS_LIGHTGBM and lgb is not None:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=15,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.75,
                reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1
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
        self.saved_at = None

    def get_feature_list(self, df):
        """Retourne la liste des features stationnaires valides présentes dans le DataFrame."""
        base_features = [
            'Returns', 'SMA_20_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 'EMA_9_Ratio',
            'SMA_50_Slope', 'SMA_200_Slope',
            'Vol_5', 'Vol_20', 'Vol_Ratio_5_20', 'Vol_Ratio_20_60', 'GK_Vol_20', 'Downside_Vol_20',
            'RSI', 'RSI_Change_5', 'MACD', 'Signal_Line', 'MACD_Hist', 'MACD_Hist_Slope',
            'BB_Percent', 'BB_Width', 'Price_ZScore_20',
            'NATR', 'ADX', 'DI_Diff', 'ER_10', 'Donchian_Pos_20',
            'Volume_Ratio', 'CLV', 'CMF_20', 'MFI_14', 'OBV_ZScore',
            'Upper_Shadow', 'Lower_Shadow', 'Gap_Open',
            'Stoch_K', 'ROC', 'Ret_5d', 'Ret_21d', 'Ret_63d', 'Dist_High_52w', 'Pos_52w',
            'VWAP_Ratio', 'Lag_1_Return', 'Lag_2_Return', 'Lag_3_Return',
            'DayOfWeek_Sin', 'DayOfWeek_Cos', 'TurnOfMonth'
        ]
        macro_features = [
            'SPY_Return', 'VIX', 'VIX_Term_Ratio', 'VIX_Change_5d', 'TNX', 'Yield_Spread', 'DXY_Return',
            'RS_vs_SPY_1d', 'RS_vs_SPY_5d', 'RS_vs_SPY_20d', 'Rolling_Beta_60'
        ]
        return [f for f in base_features + macro_features if f in df.columns]

    def _prune_collinear_features(self, X, threshold=0.95):
        """Élagage des features fortement colinéaires (|r| > threshold) pour réduire le surapprentissage."""
        if X.shape[1] <= 8 or len(X) < 30:
            return list(X.columns)
        try:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            kept = [c for c in X.columns if c not in to_drop]
            return kept if len(kept) >= 8 else list(X.columns)
        except Exception:
            return list(X.columns)

    @staticmethod
    def _compute_sample_weights(y_series, min_decay_weight=0.45):
        """
        Pondération combinée pour séries financières non-stationnaires :
        1. Décroissance temporelle exponentielle douce (donne plus de poids aux régimes récents)
        2. Rééquilibrage doux des classes (corrige le biais Triple Barrier TP 3x / SL 1.5x)
        """
        n = len(y_series)
        if n == 0:
            return None
        # Décroissance temporelle de min_decay_weight (ancien) à 1.0 (récent)
        time_w = np.geomspace(min_decay_weight, 1.0, num=n)
        y_arr = np.asarray(y_series, dtype=int)
        counts = np.bincount(y_arr, minlength=2)
        if counts[0] > 0 and counts[1] > 0:
            # Rééquilibrage modéré (racine carrée du ratio) pour éviter de sur-amplifier le bruit
            w0 = np.sqrt(n / (2.0 * counts[0]))
            w1 = np.sqrt(n / (2.0 * counts[1]))
            cls_w = np.where(y_arr == 1, w1, w0)
            weights = time_w * cls_w
        else:
            weights = time_w
        return weights / (np.mean(weights) + 1e-9)

    def _fit_with_weights(self, model, X_tr, y_tr):
        """Entraîne le modèle avec sample_weight temporel/équilibré, avec repli sûr si nécessaire."""
        sw = self._compute_sample_weights(y_tr)
        try:
            model.fit(X_tr, y_tr, sample_weight=sw)
            return model
        except TypeError:
            try:
                model.fit(X_tr, y_tr)
                return model
            except ValueError:
                pass
        except ValueError:
            pass
        # Repli propre sur l'ensemble de base si un fold temporel de calibration est mono-classe
        base_est = getattr(model, "estimator", None)
        if base_est is not None:
            try:
                base_est.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                base_est.fit(X_tr, y_tr)
            return base_est
        model.fit(X_tr, y_tr)
        return model

    def _build_ensemble(self, xgb_params=None, lgb_params=None, calibrate=False, n_samples=0, min_class_count=0):
        """Construit l'ensemble XGBoost + RF + LightGBM avec calibration temporelle (TimeSeriesSplit)."""
        xp = {
            'n_estimators': 150, 'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 5,
            'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.75, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'objective': 'binary:logistic', 'random_state': 42
        }
        if xgb_params:
            xp.update(xgb_params)
        self.xgb_model = xgb.XGBClassifier(**xp)
        self.rf_model = RandomForestClassifier(
            n_estimators=120, max_depth=5, min_samples_leaf=10, max_features='sqrt', random_state=42
        )
        estimators = [('xgb', self.xgb_model), ('rf', self.rf_model)]
        if HAS_LIGHTGBM and lgb is not None:
            lp = {
                'n_estimators': 150, 'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15,
                'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.75,
                'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': 42, 'verbose': -1
            }
            if lgb_params:
                lp.update(lgb_params)
            self.lgb_model = lgb.LGBMClassifier(**lp)
            estimators.append(('lgb', self.lgb_model))
        else:
            self.lgb_model = None

        base_ensemble = VotingClassifier(estimators=estimators, voting='soft')
        if calibrate and n_samples >= 120 and min_class_count >= 20:
            # Validation temporelle stricte (zéro fuite future dans la calibration Platt)
            from sklearn.model_selection import TimeSeriesSplit
            return CalibratedClassifierCV(estimator=base_ensemble, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
        return base_ensemble

    @staticmethod
    def _extract_feature_importances(fitted_model, n_features):
        """Extrait l'importance moyenne des features (compatible VotingClassifier et CalibratedClassifierCV)."""
        try:
            if hasattr(fitted_model, 'estimators_'):
                return np.mean([est.feature_importances_ for est in fitted_model.estimators_], axis=0)
            if hasattr(fitted_model, 'calibrated_classifiers_'):
                imps = []
                for cc in fitted_model.calibrated_classifiers_:
                    inner = getattr(cc, 'estimator', None)
                    if inner is not None and hasattr(inner, 'estimators_'):
                        imps.append(np.mean([est.feature_importances_ for est in inner.estimators_], axis=0))
                if imps:
                    return np.mean(imps, axis=0)
        except Exception:
            pass
        return np.zeros(n_features, dtype=float)

    def _optimize_hyperparams(self, X_tr, y_tr, embargo_days=5, n_trials=12):
        """
        Optimisation Bayésienne Optuna orientée finance (neg_log_loss + Purged Embargo CV)
        pour XGBoost et LightGBM avec régularisation L1/L2 et sous-échantillonnage.
        """
        xgb_best, lgb_best = None, None
        if not (HAS_OPTUNA and optuna is not None) or len(X_tr) < 80 or len(np.unique(y_tr)) < 2:
            return xgb_best, lgb_best

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import log_loss

        gap = min(embargo_days, max(0, len(X_tr) // 20))
        tscv = TimeSeriesSplit(n_splits=3, gap=gap)

        def eval_cv_logloss(model_factory):
            losses = []
            for tr_idx, va_idx in tscv.split(X_tr):
                y_f_tr, y_f_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
                if len(np.unique(y_f_tr)) < 2 or len(np.unique(y_f_va)) < 2:
                    continue
                m = model_factory()
                m.fit(X_tr.iloc[tr_idx], y_f_tr)
                probs = np.clip(m.predict_proba(X_tr.iloc[va_idx])[:, 1], 1e-6, 1.0 - 1e-6)
                losses.append(log_loss(y_f_va, probs))
            return float(np.mean(losses)) if losses else 1.0

        def obj_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 220),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
            }
            return eval_cv_logloss(lambda: xgb.XGBClassifier(**params, objective='binary:logistic', random_state=42))

        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(obj_xgb, n_trials=n_trials)
        xgb_best = study_xgb.best_params

        if HAS_LIGHTGBM and lgb is not None:
            def obj_lgb(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 60, 220),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'num_leaves': trial.suggest_int('num_leaves', 8, 31),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
                }
                return eval_cv_logloss(lambda: lgb.LGBMClassifier(**params, random_state=42, verbose=-1))

            study_lgb = optuna.create_study(direction='minimize')
            study_lgb.optimize(obj_lgb, n_trials=max(6, n_trials // 2))
            lgb_best = study_lgb.best_params

        return xgb_best, lgb_best

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
        Calcule les métriques de performance et de risque institutionnelles :
        - Annualisation adaptative (252 journalier vs 52 hebdomadaire)
        - Sortino exact (semi-déviation sous 0)
        - CAGR, Profit Factor, Calmar Ratio, Exposition %, Turnover annuel, Sharpe Buy & Hold
        """
        metrics = {}
        strat_returns = test_data['Strategy_Return'].fillna(0.0)
        periods_per_year = _infer_periods_per_year(test_data.index)
        n_periods = len(strat_returns)

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

            gross_profit = float(trades_df[trades_df['pnl'] > 0]['pnl'].sum())
            gross_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].abs().sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

            if payoff_ratio > 0 and trade_win_rate > 0:
                trade_kelly = trade_win_rate - ((1.0 - trade_win_rate) / payoff_ratio)
            else:
                trade_kelly = 0.0

            metrics['Total Trades'] = total_trades
            metrics['Trade Win Rate'] = trade_win_rate
            metrics['Payoff Ratio'] = float(payoff_ratio)
            metrics['Profit Factor'] = float(profit_factor)
            metrics['Expectancy %'] = float((trade_win_rate * avg_win) - ((1.0 - trade_win_rate) * avg_loss))
            metrics['Trade Kelly %'] = max(0.0, float(trade_kelly))
            metrics['Half Kelly %'] = min(0.05, max(0.0, float(trade_kelly) / 2.0))
            metrics['Kelly %'] = metrics['Trade Kelly %']
        else:
            winning_days = len(strat_returns[strat_returns > 0])
            total_trades_days = len(strat_returns[strat_returns != 0])
            trade_win_rate = winning_days / total_trades_days if total_trades_days > 0 else 0.0
            metrics['Total Trades'] = 0
            metrics['Trade Win Rate'] = trade_win_rate
            metrics['Payoff Ratio'] = 0.0
            metrics['Profit Factor'] = 0.0
            metrics['Trade Kelly %'] = 0.0
            metrics['Half Kelly %'] = 0.0
            metrics['Kelly %'] = 0.0

        metrics['Win Rate'] = trade_win_rate

        # 2. Drawdown Maximum & CAGR & Calmar
        cum_returns = (1.0 + strat_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        metrics['Max Drawdown'] = max_dd

        years = max(n_periods / periods_per_year, 1.0 / periods_per_year)
        final_cum = float(cum_returns.iloc[-1]) if not cum_returns.empty else 1.0
        cagr = (final_cum ** (1.0 / years) - 1.0) if final_cum > 0 else -1.0
        metrics['CAGR'] = float(cagr)
        metrics['Calmar Ratio'] = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0

        # 3. Sharpe & Sortino exact (semi-déviation par rapport à 0) & Sharpe Marché (Buy & Hold)
        mean_ret = float(strat_returns.mean())
        std_ret = float(strat_returns.std())
        metrics['Sharpe Ratio'] = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        downside_sq = np.minimum(strat_returns.values, 0.0) ** 2
        downside_dev = float(np.sqrt(np.mean(downside_sq))) if len(downside_sq) > 0 else 0.0
        metrics['Sortino Ratio'] = float(mean_ret / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else 0.0

        if 'Close' in test_data.columns and len(test_data) > 2:
            mkt_ret = test_data['Close'].pct_change().dropna()
            mkt_std = float(mkt_ret.std()) if not mkt_ret.empty else 0.0
            metrics['Market Sharpe Ratio'] = float(mkt_ret.mean() / mkt_std * np.sqrt(periods_per_year)) if mkt_std > 0 else 0.0
        else:
            metrics['Market Sharpe Ratio'] = 0.0

        # 4. Exposition (% du temps investi) & Turnover annuel
        if 'Invested' in test_data.columns and len(test_data) > 0:
            metrics['Exposure %'] = float(test_data['Invested'].mean())
        else:
            metrics['Exposure %'] = float((strat_returns != 0).mean()) if len(strat_returns) > 0 else 0.0
        metrics['Annual Turnover'] = float(metrics['Total Trades'] / years) if years > 0 else 0.0

        # 5. VaR 95% & CVaR (Expected Shortfall)
        if not strat_returns.empty:
            var_95 = float(np.percentile(strat_returns.dropna(), 5))
            metrics['VaR (95%)'] = var_95
            metrics['CVaR'] = float(strat_returns[strat_returns <= var_95].mean())
        else:
            metrics['VaR (95%)'] = 0.0
            metrics['CVaR'] = 0.0

        # 6. Probabilité de Ruine
        metrics['Probabilité de Ruine'] = self.run_monte_carlo(strat_returns.dropna().values)

        self.advanced_metrics = metrics
        return metrics

    def run_event_driven_backtest(
        self, test_data, initial_capital=100000.0,
        use_atr_stops=True, atr_sl_mult=1.5, atr_tp_mult=3.0,
        commission_rate=0.0005, slippage_rate=0.0010,
        use_breakeven_stop=True, use_trailing_stop=False,
        use_regime_filters=False, use_vol_sizing=False,
        risk_per_trade_pct=0.02,
        entry_threshold=0.55, exit_threshold=0.45
    ):
        """
        Moteur de Backtest Événementiel Strictement Causal :
        - Exécution à l'Open du signal de la veille (i-1) — zéro look-ahead bias.
        - Coupe-circuit Macro VIX > 30 basé sur VIX(i-1) (connu avant l'Open du jour i).
        - Aucune ré-entrée le même jour après une sortie / stop.
        - Mise à jour Breakeven & Trailing Stop (Chandelier Exit) en fin de séance pour effet dès i+1.
        - Commissions d'entrée et de sortie exactes, Position Sizing ATR/Vol et filtres de régime optionnels.
        """
        test_data = test_data.copy()
        cash = float(initial_capital)
        shares = 0.0

        portfolio_values = []
        daily_returns = []
        invested_flags = []
        last_val = float(initial_capital)

        has_open = 'Open' in test_data.columns
        has_high_low = ('High' in test_data.columns) and ('Low' in test_data.columns)
        has_atr = 'ATR' in test_data.columns
        has_vix = 'VIX' in test_data.columns
        has_sma200 = 'SMA_200' in test_data.columns
        has_rsi = 'RSI' in test_data.columns

        # 1. Hystérésis de Probabilité (seuils paramétrables)
        signals = []
        curr_sig = 0
        probs = test_data['Prob'].values if 'Prob' in test_data.columns else [0.5] * len(test_data)
        for p in probs:
            if p > entry_threshold:
                curr_sig = 1
            elif p < exit_threshold:
                curr_sig = 0
            signals.append(curr_sig)
        test_data['Signal'] = signals

        # Coupe-circuit Macro VIX sur le signal de fin de journée (appliqué le lendemain à l'Open)
        if has_vix:
            test_data.loc[test_data['VIX'] > 30, 'Signal'] = 0

        closed_trades = []
        entry_price = 0.0
        entry_commission = 0.0
        entry_atr = 0.0
        entry_date = None
        entry_index = None
        current_sl = 0.0
        current_tp = 0.0
        highest_since_entry = 0.0

        for i in range(len(test_data)):
            row = test_data.iloc[i]
            prev_row = test_data.iloc[i - 1] if i > 0 else None
            date = test_data.index[i]

            # Décisions strictement issues de la veille (i-1)
            raw_signal = 0 if i == 0 else int(test_data['Signal'].iloc[i - 1])
            prev_vix = float(prev_row['VIX']) if (prev_row is not None and has_vix and not pd.isna(prev_row['VIX'])) else 18.0
            current_open = float(row['Open']) if has_open else float(row['Close'])
            current_close = float(row['Close'])
            current_high = float(row['High']) if has_high_low else current_close
            current_low = float(row['Low']) if has_high_low else current_close

            if prev_row is not None and has_atr and not pd.isna(prev_row['ATR']):
                prev_atr = float(prev_row['ATR'])
            elif has_atr and not pd.isna(row['ATR']):
                prev_atr = float(row['ATR'])
            else:
                prev_atr = current_open * 0.02

            exited_today = False

            # --- VÉRIFICATION DES POSITIONS OUVERTES ---
            if shares > 0:
                exit_price = None
                exit_reason = None

                # A. Coupe-circuit Macro VIX > 30 connu avant l'ouverture (VIX de la veille i-1)
                if has_vix and prev_vix > 30:
                    exit_price = current_open * (1.0 - slippage_rate)
                    exit_reason = "VIX > 30 (Circuit Breaker)"
                # B. Signal ML inversé connu avant l'ouverture (Prob(i-1) < 0.45)
                elif raw_signal == 0:
                    exit_price = current_open * (1.0 - slippage_rate)
                    exit_reason = "Signal Exit (< 0.45)"
                # C. Stop-Loss / Breakeven / Trailing Stop ATR touché en séance (niveaux fixés avant séance)
                elif use_atr_stops and current_low <= current_sl:
                    exit_price = min(current_open, current_sl) * (1.0 - slippage_rate)
                    if use_trailing_stop and current_sl > entry_price:
                        exit_reason = "Trailing Stop ATR (Chandelier)"
                    elif current_sl >= entry_price:
                        exit_reason = "Breakeven Stop (0 Risque)"
                    else:
                        exit_reason = f"Stop Loss ({atr_sl_mult:.1f} ATR)"
                # D. Take-Profit ATR fixe (uniquement si Trailing Stop non activé)
                elif use_atr_stops and (not use_trailing_stop) and current_high >= current_tp:
                    exit_price = max(current_open, current_tp) * (1.0 - slippage_rate)
                    exit_reason = f"Take Profit ({atr_tp_mult:.1f} ATR)"

                # Exécution de la sortie
                if exit_price is not None:
                    proceeds = shares * exit_price
                    exit_commission = proceeds * commission_rate
                    cash += proceeds - exit_commission

                    cost_basis = (entry_price * shares) + entry_commission
                    pnl = (proceeds - exit_commission) - cost_basis
                    pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0
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
                    entry_commission = 0.0
                    entry_atr = 0.0
                    entry_date = None
                    entry_index = None
                    current_sl = 0.0
                    current_tp = 0.0
                    highest_since_entry = 0.0
                    exited_today = True
                else:
                    # Mise à jour causale en FIN de séance (effective dès la séance suivante i+1)
                    highest_since_entry = max(highest_since_entry, current_high)
                    if use_atr_stops and use_breakeven_stop and highest_since_entry >= entry_price + (atr_sl_mult * entry_atr):
                        current_sl = max(current_sl, entry_price)
                    if use_atr_stops and use_trailing_stop and highest_since_entry >= entry_price + (atr_sl_mult * entry_atr):
                        chandelier_sl = highest_since_entry - (2.0 * entry_atr)
                        current_sl = max(current_sl, chandelier_sl)

            # --- VÉRIFICATION DES NOUVEAUX ACHATS (jamais le même jour qu'une sortie) ---
            if raw_signal == 1 and shares == 0 and not exited_today:
                regime_ok = True
                if use_regime_filters and prev_row is not None:
                    if has_sma200 and not pd.isna(prev_row['SMA_200']) and float(prev_row['Close']) < float(prev_row['SMA_200']):
                        regime_ok = False
                    if has_rsi and not pd.isna(prev_row['RSI']) and float(prev_row['RSI']) >= 75.0:
                        regime_ok = False

                if regime_ok and cash > 10.0:
                    exec_price = current_open * (1.0 + slippage_rate)
                    stop_dist = max(atr_sl_mult * prev_atr, exec_price * 0.005)

                    if use_vol_sizing:
                        # Risque fixe (ex: 2% du capital par stop ATR), modulé si VIX élevé (20-30)
                        risk_budget = cash * risk_per_trade_pct
                        if has_vix and 20.0 < prev_vix <= 30.0:
                            risk_budget *= 0.75
                        target_shares = risk_budget / stop_dist
                        max_investable = cash / (1.0 + commission_rate)
                        investable = min(target_shares * exec_price, max_investable)
                    else:
                        investable = cash / (1.0 + commission_rate)

                    if investable > 10.0:
                        shares = investable / exec_price
                        entry_commission = investable * commission_rate
                        cash -= (shares * exec_price + entry_commission)

                        entry_price = exec_price
                        entry_atr = prev_atr
                        entry_date = date
                        entry_index = i
                        highest_since_entry = max(current_high, entry_price)
                        current_sl = entry_price - (atr_sl_mult * entry_atr)
                        current_tp = entry_price + (atr_tp_mult * entry_atr)

            # Évaluation Mark-to-Market de fin de séance
            current_val = cash + (shares * current_close)
            portfolio_values.append(current_val)
            invested_flags.append(1.0 if shares > 0 else 0.0)
            ret = (current_val - last_val) / last_val if last_val > 0 else 0.0
            daily_returns.append(ret)
            last_val = current_val

        test_data['Event_Portfolio_Value'] = portfolio_values
        test_data['Strategy_Return'] = daily_returns
        test_data['Invested'] = invested_flags
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
        progress_callback=None,
        calibrate=False, embargo_days=5, retrain_final=True,
        prune_collinear=True, use_trailing_stop=False,
        use_regime_filters=False, use_vol_sizing=False
    ):
        """
        Entraîne le modèle avec ou sans Walk-Forward Analysis (WFA), Purged Embargo,
        Optuna (neg_log_loss + régularisation), calibration probabiliste et ré-entraînement final jusqu'à t-1.
        """
        raw_features = self.get_feature_list(data)
        valid_train = data.dropna(subset=['Target'])
        if prune_collinear:
            features = self._prune_collinear_features(valid_train[raw_features], threshold=0.95)
        else:
            features = list(raw_features)
        self.features = list(features)

        X = valid_train[features]
        y = valid_train['Target'].astype(int)

        if not use_wfa:
            split = int(len(X) * 0.8)
            gap = min(embargo_days, max(0, split // 15))
            train_end_idx = max(20, split - gap)
            X_train, X_test = X.iloc[:train_end_idx], X.iloc[split:]
            y_train, y_test = y.iloc[:train_end_idx], y.iloc[split:]

            if len(np.unique(y_train)) < 2:
                y_train = y_train.copy()
                y_train.iloc[0] = 1 - int(y_train.iloc[0])

            xgb_best, lgb_best = (None, None)
            if optimize:
                xgb_best, lgb_best = self._optimize_hyperparams(X_train, y_train, embargo_days=embargo_days)

            min_cls = int(np.bincount(y_train).min()) if len(np.unique(y_train)) >= 2 else 0
            eval_model = self._build_ensemble(xgb_best, lgb_best, calibrate=calibrate, n_samples=len(X_train), min_class_count=min_cls)
            eval_model = self._fit_with_weights(eval_model, X_train, y_train)

            # Importances des variables
            importances = self._extract_feature_importances(eval_model, len(features))
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': importances}
            ).sort_values(by='Importance', ascending=True)

            # Backtest strictement out-of-sample sur le jeu de test
            test_data = valid_train.iloc[split:].copy()
            clean_test = test_data[features]
            test_data['Prob'] = eval_model.predict_proba(clean_test)[:, 1]
            test_data = self.run_event_driven_backtest(
                test_data,
                use_trailing_stop=use_trailing_stop,
                use_regime_filters=use_regime_filters,
                use_vol_sizing=use_vol_sizing
            )

            self.backtest_results = test_data
            self.accuracy = accuracy_score(y_test, eval_model.predict(X_test))

            # Ré-entraînement final jusqu'à t-1 pour la prédiction en direct
            if retrain_final and len(X) >= 50 and len(np.unique(y)) >= 2:
                min_cls_full = int(np.bincount(y).min())
                prod_model = self._build_ensemble(xgb_best, lgb_best, calibrate=calibrate, n_samples=len(X), min_class_count=min_cls_full)
                self.model = self._fit_with_weights(prod_model, X, y)
            else:
                self.model = eval_model

            self.is_trained = True
            return features

        else:
            # --- WALK FORWARD ANALYSIS (WFA) DYNAMIQUE AVEC PURGED EMBARGO ---
            window_map = {"2Y": pd.DateOffset(years=2), "5Y": pd.DateOffset(years=5), "7Y": pd.DateOffset(years=7)}
            step_map = {"1M": pd.DateOffset(months=1), "2M": pd.DateOffset(months=2), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}

            train_window = window_map.get(wfa_train_window, pd.DateOffset(years=5))
            step_size = step_map.get(wfa_step, pd.DateOffset(months=6))
            embargo_offset = pd.Timedelta(days=int(embargo_days))

            last_ts = pd.Timestamp(valid_train.index[-1])
            first_ts = pd.Timestamp(valid_train.index[0])

            if wfa_end_date is None:
                wfa_end_date = last_ts.date()
            if wfa_start_date is None:
                wfa_start_date = (first_ts + train_window).date()

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
            xgb_best, lgb_best = (None, None)
            last_fitted_m = None

            while current_date < end_date:
                train_start = current_date - train_window
                train_cutoff = current_date - embargo_offset
                test_end = current_date + step_size

                mask_train = (valid_train.index >= train_start) & (valid_train.index < train_cutoff)
                mask_test = (valid_train.index >= current_date) & (valid_train.index < test_end)

                X_tr, y_tr = X[mask_train], y[mask_train]
                X_te, y_te_slice = X[mask_test], y[mask_test]

                if len(X_tr) < 50 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                    current_date += step_size
                    continue

                if optimize and xgb_best is None:
                    xgb_best, lgb_best = self._optimize_hyperparams(X_tr, y_tr, embargo_days=embargo_days, n_trials=8)

                min_cls = int(np.bincount(y_tr).min())
                m = self._build_ensemble(xgb_best, lgb_best, calibrate=calibrate, n_samples=len(X_tr), min_class_count=min_cls)
                m = self._fit_with_weights(m, X_tr, y_tr)
                last_fitted_m = m

                prob = m.predict_proba(X_te)[:, 1]
                pred = m.predict(X_te)

                test_slice = valid_train[mask_test].copy()
                test_slice['Prob'] = prob
                all_test_data.append(test_slice)

                y_test_all.extend(y_te_slice.values)
                y_pred_all.extend(pred)

                imp = self._extract_feature_importances(m, len(features))
                all_importances.append(imp)

                current_date += step_size
                if progress_callback and start_days > 0:
                    prog = min(1.0, 1.0 - ((end_date - current_date).days / start_days))
                    progress_callback(prog)

            if len(all_test_data) == 0:
                raise ValueError(
                    f"Historique insuffisant pour la fenêtre d'entraînement WFA ({wfa_train_window}). "
                    f"La période d'historique choisie ne contient pas assez d'échantillons pour constituer les sous-fenêtres. "
                    f"Veuillez choisir une période d'historique plus étendue (ex: 5y ou 10y) ou raccourcir la fenêtre WFA (ex: 2Y)."
                )

            final_test_data = pd.concat(all_test_data)
            self.feature_importances = pd.DataFrame(
                {'Feature': features, 'Importance': np.mean(all_importances, axis=0)}
            ).sort_values(by='Importance', ascending=True)

            final_test_data = self.run_event_driven_backtest(
                final_test_data,
                use_trailing_stop=use_trailing_stop,
                use_regime_filters=use_regime_filters,
                use_vol_sizing=use_vol_sizing
            )
            self.backtest_results = final_test_data
            self.accuracy = accuracy_score(y_test_all, y_pred_all) if len(y_test_all) > 0 else 0.0

            # Ré-entraînement final sur la fenêtre la plus récente jusqu'à t-1
            recent_mask = valid_train.index >= (last_ts - train_window)
            X_rec, y_rec = X[recent_mask], y[recent_mask]
            if retrain_final and len(X_rec) >= 50 and len(np.unique(y_rec)) >= 2:
                min_cls_rec = int(np.bincount(y_rec).min())
                prod_m = self._build_ensemble(xgb_best, lgb_best, calibrate=calibrate, n_samples=len(X_rec), min_class_count=min_cls_rec)
                self.model = self._fit_with_weights(prod_m, X_rec, y_rec)
            else:
                self.model = last_fitted_m

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
        X_input = last_row.reindex(columns=features_expected, fill_value=0.0)
        prob = float(self.model.predict_proba(X_input)[0][1])

        if 'VIX' in last_row.columns:
            vix_val = float(last_row['VIX'].iloc[0]) if isinstance(last_row['VIX'], pd.Series) else float(last_row['VIX'])
            if vix_val > 30:
                return 0.0
        return prob

    def save(self, filepath):
        """Sauvegarde l'état du modèle via joblib (avec horodatage pour TTL) ou pickle en repli."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        self.saved_at = datetime.now().isoformat()
        state = {
            'model': self.model,
            'accuracy': self.accuracy,
            'feature_importances': self.feature_importances,
            'backtest_results': self.backtest_results,
            'is_trained': self.is_trained,
            'features': getattr(self, 'features', []),
            'advanced_metrics': getattr(self, 'advanced_metrics', {}),
            'closed_trades': getattr(self, 'closed_trades', []),
            'saved_at': self.saved_at,
        }
        if HAS_JOBLIB and joblib is not None:
            joblib.dump(state, filepath, compress=3)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

    @staticmethod
    def load(filepath):
        """Charge un modèle persisté (compatible joblib et anciens fichiers pickle)."""
        state = None
        if HAS_JOBLIB and joblib is not None:
            try:
                state = joblib.load(filepath)
            except Exception:
                state = None
        if state is None:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        trader = MLTrader()
        trader.model = state['model']
        trader.accuracy = state['accuracy']
        trader.feature_importances = state['feature_importances']
        trader.backtest_results = state['backtest_results']
        trader.is_trained = state['is_trained']
        trader.features = state.get('features', [])
        trader.advanced_metrics = state.get('advanced_metrics', {})
        trader.closed_trades = state.get('closed_trades', [])
        trader.saved_at = state.get('saved_at', None)
        return trader


# =============================================================================
# 5. ALLOCATION BAYÉSIENNE DE BLACK-LITTERMAN
# =============================================================================

def get_black_litterman_weights(returns_df, market_weights, views_dict, tau=0.05, risk_aversion=2.5, periods_per_year=None):
    """
    Calcule les poids de portefeuille optimaux par le modèle de Black-Litterman.
    returns_df : rendements historiques des actions sélectionnées
    market_weights : prior d'équilibre (ex: Parité des Risques 1/Vol)
    views_dict : vues subjectives d'excès de rendement issues de l'IA {ticker: excess_return}
                 (ajoutées au prior d'équilibre Pi afin qu'une vue neutre 0.0 préserve Pi)
    """
    try:
        tickers = list(returns_df.columns)
        N = len(tickers)
        if N == 0:
            return {}
        if N == 1:
            return {tickers[0]: 1.0}

        ppy = float(periods_per_year) if periods_per_year is not None else _infer_periods_per_year(returns_df.index)
        Sigma = returns_df.cov().values * ppy
        if np.isnan(Sigma).any():
            return market_weights
        # Régularisation diagonale minimale pour garantir la stricte inversibilité
        Sigma += np.eye(N) * 1e-6

        W_mkt = np.array([market_weights.get(t, 1.0 / N) for t in tickers])
        Pi = risk_aversion * Sigma.dot(W_mkt)

        P = np.eye(N)
        excess_views = np.array([views_dict.get(t, 0.0) for t in tickers])

        if np.all(excess_views == 0.0):
            return market_weights

        # Correction Black-Litterman : Q est le rendement attendu absolu (Pi + excès IA)
        Q = Pi + excess_views
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
    commission_per_leg=1.0, iv_vrp_mult=1.15, progress_callback=None
):
    """
    Simulateur de stratégies d'options systématiques avec gestion réaliste des frictions :
    - Volatilité strictement causale (sans bfill) + prime de risque de volatilité (VRP).
    - Décompte réel en jours calendaires jusqu'à l'échéance (cohérent avec T = DTE / 365).
    """
    returns = closes.pct_change()
    ppy = _infer_periods_per_year(closes.index)
    hist_vol = (returns.rolling(window=30, min_periods=5).std() * np.sqrt(ppy)).fillna(0.20)
    r = 0.05
    
    cash = float(capital)
    equity_curve = []
    trades = []
    current_position = None
    total_days = len(closes)
    
    for i, (date, spot) in enumerate(closes.items()):
        if progress_callback and i % 25 == 0:
            progress_callback(i / total_days)
            
        sigma = max(float(hist_vol.iloc[i]) * float(iv_vrp_mult), 0.05)
        
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
            entry_dt = current_position.get('entry_date')
            if hasattr(date, 'to_pydatetime') and hasattr(entry_dt, 'to_pydatetime'):
                elapsed_cal_days = max(1, (date - entry_dt).days)
            elif hasattr(date - entry_dt, 'days'):
                elapsed_cal_days = max(1, (date - entry_dt).days)
            else:
                elapsed_cal_days = int(round(current_position['days_passed'] * (365.0 / 252.0)))
            days_remaining = target_dte - elapsed_cal_days
            
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

