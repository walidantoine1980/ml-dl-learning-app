"""
===============================================================================
MOTEUR DE TRADING AUTOMATIQUE EN BOUCLE (AUTOPILOT ENGINE)
XGBoost Stock Predictor Pro - Clean Architecture
===============================================================================
Module Python pur (sans aucune dépendance à Streamlit) orchestrant :
1. La sélection et fusion multi-fichiers CSV de trackers (WatchList/)
2. Le cycle autonome en boucle continue après apprentissage (WFA ou standard) :
   - Audit des positions ouvertes (Breakeven +1R, Trailing Stop ATR, Take-Profit,
     Stop-Loss, Signal IA baissier) -> Vente automatique selon rentabilité
   - Classement des opportunités d'achat (Score composite : conviction IA,
     Sharpe/Profit Factor d'apprentissage, filtres VIX / RSI / Tendance)
   - Achat automatique avec allocation maîtrisée et réserve de cash
   - Répétition en boucle (Achat -> Vente -> Ré-achat -> Revente...)
3. L'interruption immédiate et propre (bouton Stop / Pause / Liquidation totale)

Note : Ce module n'altère aucune stratégie de `quant_core.py` ; il consomme
uniquement les modèles entraînés (`MLTrader`) et les fonctions d'exécution.
===============================================================================
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_core import (
    MLTrader,
    PORTFOLIO_FILE,
    add_features,
    convert_google_to_yahoo_ticker,
    evaluate_multi_regime_triggers,
    execute_equity_trade,
    get_company_name_from_yahoo,
    get_macro_data,
    join_macro_data,
    load_portfolio,
    save_portfolio,
)
from tickers_db import parse_and_validate_watchlist_csv

def _get_persistent_runtime_dirs() -> List[str]:
    """Retourne les dossiers persistants (Docker /app/data + profil utilisateur ~/.xgboost_trader)."""
    dirs: List[str] = []
    env_dir = os.environ.get("XGBOOST_PORTFOLIO_DIR", "").strip()
    if env_dir:
        dirs.append(os.path.abspath(env_dir))
    try:
        user_home = os.path.expanduser("~")
        if user_home and user_home != "~":
            dirs.append(os.path.join(user_home, ".xgboost_trader"))
    except Exception:
        pass
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.join(base_dir, "data"))

    unique: List[str] = []
    seen = set()
    for d in dirs:
        norm = os.path.normcase(os.path.abspath(d))
        if norm not in seen:
            seen.add(norm)
            unique.append(os.path.abspath(d))
    return unique


def _get_persistent_runtime_file(filename: str) -> str:
    """
    Résout le chemin principal persistant pour la configuration et l'état du mode automatique.
    Priorité : XGBOOST_PORTFOLIO_DIR (ex: /app/data en Docker), puis ~/.xgboost_trader, puis ./data/.
    """
    for target_dir in _get_persistent_runtime_dirs():
        try:
            os.makedirs(target_dir, exist_ok=True)
            return os.path.join(target_dir, filename)
        except Exception:
            continue
    return filename


AUTO_TRADER_STATE_FILE = _get_persistent_runtime_file("auto_trader_state.json")
AUTO_TRADER_CONFIG_FILE = _get_persistent_runtime_file("auto_trader_config.json")


# =============================================================================
# 1. UTILITAIRES MULTI-CSV WATCHLISTS
# =============================================================================

def get_watchlists_dir(watchlists_dir: Optional[str] = None) -> str:
    """Retourne le chemin absolu du dossier WatchList/."""
    if watchlists_dir:
        return watchlists_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "WatchList")


def list_watchlist_csv_files(watchlists_dir: Optional[str] = None) -> List[str]:
    """Liste tous les fichiers .csv disponibles dans le dossier WatchList/ triés par nom."""
    wdir = get_watchlists_dir(watchlists_dir)
    if not os.path.isdir(wdir):
        return []
    files = [
        f for f in os.listdir(wdir)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(wdir, f))
    ]
    return sorted(files, key=lambda s: s.lower())


def load_tickers_from_csv_files(
    csv_filenames: List[str],
    watchlists_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Charge, valide et fusionne plusieurs fichiers CSV de trackers depuis WatchList/.
    Déduplique automatiquement les symboles Yahoo Finance tout en conservant
    la traçabilité du fichier source.
    """
    wdir = get_watchlists_dir(watchlists_dir)
    seen_symbols: set[str] = set()
    entries: List[Dict[str, str]] = []
    file_reports: List[Dict[str, Any]] = []
    errors: List[str] = []

    for fname in csv_filenames:
        fpath = fname if os.path.isabs(fname) else os.path.join(wdir, fname)
        short_name = os.path.basename(fpath)
        if not os.path.isfile(fpath):
            msg = f"Fichier introuvable : {short_name}"
            errors.append(msg)
            file_reports.append({"file": short_name, "valid": False, "count": 0, "error": msg})
            continue

        try:
            with open(fpath, "rb") as f:
                raw_bytes = f.read()
            res = parse_and_validate_watchlist_csv(raw_bytes, short_name)
            if not res.get("is_valid"):
                err_txt = " ; ".join(res.get("errors", [])) or "Format CSV non conforme"
                errors.append(f"{short_name}: {err_txt}")
                file_reports.append({"file": short_name, "valid": False, "count": 0, "error": err_txt})
                continue

            added = 0
            for item in res.get("entries", []):
                sym = convert_google_to_yahoo_ticker(item["symbol"])
                if not sym or sym in seen_symbols:
                    continue
                seen_symbols.add(sym)
                entries.append({
                    "symbol": sym,
                    "name": item.get("name", sym),
                    "display_name": item.get("display_name", f"{sym} ({sym})"),
                    "exchange": item.get("exchange", ""),
                    "source_file": short_name,
                })
                added += 1

            file_reports.append({
                "file": short_name,
                "valid": True,
                "count": len(res.get("entries", [])),
                "added_unique": added,
                "format": res.get("format_detected", ""),
            })
        except Exception as exc:
            err_txt = f"{short_name}: {exc}"
            errors.append(err_txt)
            file_reports.append({"file": short_name, "valid": False, "count": 0, "error": str(exc)})

    return {
        "symbols": [e["symbol"] for e in entries],
        "entries": entries,
        "file_reports": file_reports,
        "errors": errors,
        "total_unique": len(entries),
    }


# =============================================================================
# 2. CONFIGURATION DU MODE AUTOMATIQUE (PARAMÈTRES EXPERT)
# =============================================================================

@dataclass
class AutoTraderConfig:
    """Paramètres complets du moteur de trading automatique."""
    tickers: List[str] = field(default_factory=list)
    csv_sources: List[str] = field(default_factory=list)
    # Cadence et capacité du portefeuille
    scan_interval_sec: int = 3600
    max_open_positions: int = 10
    position_size_pct: float = 15.0
    min_cash_reserve_pct: float = 10.0
    # Seuils de décision IA (consomment la prédiction du modèle sans toucher aux stratégies)
    min_buy_prob: float = 0.52
    sell_prob_threshold: float = 0.50
    # Critères de rentabilité et gestion dynamique du risque
    take_profit_pct: float = 3.0
    stop_loss_pct: float = 4.0
    use_live_intraday_price: bool = True
    use_atr_stops: bool = True
    use_trailing_stop: bool = True
    trailing_atr_mult: float = 1.25
    trailing_stop_pct: float = 1.5
    use_breakeven_stop: bool = True
    use_partial_take_profit: bool = True
    partial_take_profit_pct: float = 1.5
    partial_take_profit_ratio: float = 0.50
    # Filtres de protection & coupe-circuits institutionnels
    use_vix_breaker: bool = True
    vix_max_threshold: float = 30.0
    use_rsi_shield: bool = True
    use_sma200_filter: bool = False
    # Point 1 : Filtre Qualité WFA (n'achète que les modèles rentables en apprentissage)
    use_wfa_quality_filter: bool = True
    min_wfa_profit_factor: float = 0.90
    min_wfa_sharpe: float = 0.0
    # Point 2 : Sizing adaptatif au régime de marché (SPY & VIX)
    use_adaptive_regime_sizing: bool = True
    defensive_size_multiplier: float = 0.50
    max_drawdown_pct: float = 12.0
    cooldown_sec: int = 3600
    # Mode simulation (Dry-Run) vs Paper Trading réel (portfolio.json)
    dry_run: bool = False
    # Paramètres de données / apprentissage (informatifs pour cohérence)
    period: str = "5y"
    interval: str = "1d"
    labeling_mode: str = "triple_barrier"
    learning_mode: str = "Walk-Forward Analysis (WFA)"
    models_dir: str = "models"
    portfolio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoTraderConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (data or {}).items() if k in valid_fields}
        return cls(**filtered)


def save_auto_trader_config(config: AutoTraderConfig, filepath: str = AUTO_TRADER_CONFIG_FILE) -> None:
    """Sauvegarde atomique multi-emplacements de la configuration du mode automatique."""
    targets = [filepath]
    if filepath == AUTO_TRADER_CONFIG_FILE:
        for d in _get_persistent_runtime_dirs():
            p = os.path.join(d, "auto_trader_config.json")
            if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(filepath)):
                targets.append(p)
    payload = config.to_dict()
    for target in targets:
        try:
            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{target}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, target)
        except Exception:
            pass


def load_auto_trader_config(filepath: str = AUTO_TRADER_CONFIG_FILE) -> AutoTraderConfig:
    """Charge la configuration sauvegardée la plus récente parmi les emplacements persistants."""
    candidates = [filepath]
    if filepath == AUTO_TRADER_CONFIG_FILE:
        for d in _get_persistent_runtime_dirs():
            candidates.append(os.path.join(d, "auto_trader_config.json"))
        candidates.append("auto_trader_config.json")

    best_data = None
    best_mtime = -1.0
    for path in candidates:
        if path and os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and mtime >= best_mtime:
                    best_mtime = mtime
                    best_data = data
            except Exception:
                continue
    if best_data is not None:
        return AutoTraderConfig.from_dict(best_data)
    return AutoTraderConfig()


# =============================================================================
# 3. MOTEUR AUTONOME EN BOUCLE CONTINUE (THREAD-SAFE)
# =============================================================================

class AutoTraderEngine:
    """
    Moteur autonome d'exécution en boucle continue :
    1. Vérifie les coupe-circuits (VIX, Drawdown Max)
    2. Surveille et vend automatiquement les positions selon rentabilité & risque
    3. Sélectionne et achète automatiquement les meilleures actions scorées par l'IA
    4. Boucle jusqu'à interruption explicite via `stop()`
    """

    def __init__(
        self,
        data_fetcher: Optional[Callable[[str, str, str], Optional[pd.DataFrame]]] = None,
        macro_fetcher: Optional[Callable[[str, str], Optional[pd.DataFrame]]] = None,
        price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        state_file: str = AUTO_TRADER_STATE_FILE,
    ) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = paused
        self._thread: Optional[threading.Thread] = None

        self._data_fetcher = data_fetcher or self._default_fetch_ohlcv
        self._macro_fetcher = macro_fetcher or self._default_fetch_macro
        self._price_fetcher = price_fetcher or self._default_fetch_price
        self._state_file = state_file

        self.config: AutoTraderConfig = AutoTraderConfig()
        self._trained_models: Dict[str, MLTrader] = {}
        self._model_metrics: Dict[str, Dict[str, Any]] = {}
        self._last_sell_ts: Dict[str, float] = {}

        # Métriques temps réel de la session autonome
        self._is_running: bool = False
        self._started_at: Optional[str] = None
        self._stopped_at: Optional[str] = None
        self._last_cycle_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._cycle_count: int = 0
        self._total_buys: int = 0
        self._total_sells: int = 0
        self._realized_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._drawdown_pct: float = 0.0
        self._drawdown_halted: bool = False
        self._last_vix: Optional[float] = None
        self._market_regime: str = "HAUSSIER CALME (100%)"
        self._effective_position_pct: float = 15.0
        self._last_rankings: List[Dict[str, Any]] = []
        self._logs: List[Dict[str, Any]] = []
        self._trade_events: List[Dict[str, Any]] = []
        self._equity_history: List[Dict[str, Any]] = []
        # Restaure automatiquement les transactions, logs et métriques après redémarrage/régénération
        self._restore_state()

    # -------------------------------------------------------------------------
    # Fetchers par défaut (yfinance + quant_core)
    # -------------------------------------------------------------------------
    @staticmethod
    def _default_fetch_ohlcv(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        import yfinance as yf

        df_raw = yf.download(symbol, period=period, interval=interval, progress=False)
        if df_raw is None or df_raw.empty:
            return None
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
        return df_raw

    @staticmethod
    def _default_fetch_macro(period: str, interval: str) -> Optional[pd.DataFrame]:
        try:
            return get_macro_data(period=period, interval=interval)
        except Exception:
            return None

    @staticmethod
    def _default_fetch_price(symbol: str) -> Optional[float]:
        import yfinance as yf

        try:
            tk = yf.Ticker(symbol)
            fast_info = getattr(tk, "fast_info", None)
            if fast_info is not None and hasattr(fast_info, "last_price"):
                lp = fast_info.last_price
                if lp is not None and float(lp) > 0:
                    return float(lp)
            hist = tk.history(period="2d")
            if hist is not None and not hist.empty:
                val = float(hist["Close"].dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception:
            return None
        return None

    # -------------------------------------------------------------------------
    # Journalisation & Persistance d'état
    # -------------------------------------------------------------------------
    def _log(self, level: str, message: str, ticker: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "ticker": ticker,
            "message": message,
        }
        if extra:
            entry.update(extra)
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > 300:
                self._logs = self._logs[-300:]

    def _record_trade_event(
        self,
        action: str,
        ticker: str,
        qty: float,
        price: float,
        reason: str,
        pnl: float = 0.0,
        dry_run: bool = False,
    ) -> None:
        event = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": f"{action} (DRY-RUN)" if dry_run else action,
            "ticker": ticker,
            "qty": round(float(qty), 4),
            "price": round(float(price), 4),
            "total": round(float(qty) * float(price), 2),
            "pnl": round(float(pnl), 2),
            "reason": reason,
        }
        with self._lock:
            self._trade_events.append(event)
            if len(self._trade_events) > 500:
                self._trade_events = self._trade_events[-500:]

    def _restore_state(self) -> None:
        """Restaure l'historique des transactions, logs et métriques depuis le stockage persistant."""
        if not self._state_file:
            return
        candidates = [self._state_file]
        if self._state_file == AUTO_TRADER_STATE_FILE:
            for d in _get_persistent_runtime_dirs():
                candidates.append(os.path.join(d, "auto_trader_state.json"))
            candidates.append("auto_trader_state.json")

        best_snap = None
        best_score = (-1, -1.0)
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                mtime = os.path.getmtime(path)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                ev_count = len(data.get("trade_events", []) or [])
                score = (ev_count, mtime)
                if score > best_score:
                    best_score = score
                    best_snap = data
            except Exception:
                continue

        if not best_snap:
            return

        with self._lock:
            self._cycle_count = int(best_snap.get("cycle_count", 0) or 0)
            self._total_buys = int(best_snap.get("total_buys", 0) or 0)
            self._total_sells = int(best_snap.get("total_sells", 0) or 0)
            self._realized_pnl = float(best_snap.get("realized_pnl", 0.0) or 0.0)
            self._peak_equity = float(best_snap.get("peak_equity", 0.0) or 0.0)
            self._current_equity = float(best_snap.get("current_equity", 0.0) or 0.0)
            self._drawdown_pct = float(best_snap.get("drawdown_pct", 0.0) or 0.0)
            self._last_cycle_at = best_snap.get("last_cycle_at")
            self._last_vix = best_snap.get("last_vix")
            if best_snap.get("market_regime"):
                self._market_regime = str(best_snap["market_regime"])
            if best_snap.get("effective_position_pct") is not None:
                self._effective_position_pct = float(best_snap["effective_position_pct"])
            if isinstance(best_snap.get("last_rankings"), list):
                self._last_rankings = list(best_snap["last_rankings"])[-50:]
            if isinstance(best_snap.get("logs"), list):
                self._logs = list(best_snap["logs"])[-300:]
            if isinstance(best_snap.get("trade_events"), list):
                self._trade_events = list(best_snap["trade_events"])[-500:]
            if isinstance(best_snap.get("equity_history"), list):
                self._equity_history = list(best_snap["equity_history"])[-500:]

    def _persist_state(self) -> None:
        """Sauvegarde atomique multi-emplacements des transactions et de l'état Autopilot."""
        if not self._state_file:
            return
        targets = [self._state_file]
        if self._state_file == AUTO_TRADER_STATE_FILE:
            for d in _get_persistent_runtime_dirs():
                p = os.path.join(d, "auto_trader_state.json")
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(self._state_file)):
                    targets.append(p)
        snap = self.get_status()
        for target in targets:
            try:
                parent = os.path.dirname(os.path.abspath(target))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                tmp_path = f"{target}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(snap, f, indent=2, ensure_ascii=False, default=str)
                os.replace(tmp_path, target)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Accès portefeuille (respecte portfolio_path si fourni pour les tests)
    # -------------------------------------------------------------------------
    def _load_pf(self) -> Dict[str, Any]:
        if self.config.portfolio_path:
            return load_portfolio(filepath=self.config.portfolio_path)
        return load_portfolio()

    def _save_pf(self, pf: Dict[str, Any]) -> None:
        if self.config.dry_run:
            return
        if self.config.portfolio_path:
            save_portfolio(pf, filepath=self.config.portfolio_path)
        else:
            save_portfolio(pf)

    # -------------------------------------------------------------------------
    # Gestion des modèles entraînés
    # -------------------------------------------------------------------------
    def register_models(
        self,
        trained_models: Dict[str, MLTrader],
        model_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Enregistre en mémoire les modèles déjà entraînés (WFA ou Standard)."""
        with self._lock:
            for sym, model in (trained_models or {}).items():
                if model is not None and getattr(model, "is_trained", False):
                    self._trained_models[sym] = model
                    if hasattr(model, "advanced_metrics") and isinstance(model.advanced_metrics, dict):
                        self._model_metrics.setdefault(sym, {}).update(model.advanced_metrics)
                        self._model_metrics[sym]["accuracy"] = getattr(model, "accuracy", 0.5)
            if model_metrics:
                for sym, m in model_metrics.items():
                    self._model_metrics.setdefault(sym, {}).update(m)

    def _get_trader(self, symbol: str) -> Optional[MLTrader]:
        with self._lock:
            trader = self._trained_models.get(symbol)
            if trader is not None and getattr(trader, "is_trained", False):
                return trader

        # Repli : chargement depuis models/{symbol}_model.pkl si sauvegardé après apprentissage
        model_path = os.path.join(self.config.models_dir, f"{symbol}_model.pkl")
        if os.path.exists(model_path):
            try:
                loaded = MLTrader.load(model_path)
                if loaded is not None and getattr(loaded, "is_trained", False):
                    with self._lock:
                        self._trained_models[symbol] = loaded
                        if hasattr(loaded, "advanced_metrics") and isinstance(loaded.advanced_metrics, dict):
                            self._model_metrics.setdefault(symbol, {}).update(loaded.advanced_metrics)
                            self._model_metrics[symbol]["accuracy"] = getattr(loaded, "accuracy", 0.5)
                    return loaded
            except Exception:
                return None
        return None

    # -------------------------------------------------------------------------
    # Scoring de sélection des meilleures actions à acheter
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_candidate_score(
        prob_buy: float,
        metrics: Optional[Dict[str, Any]] = None,
        triggers: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calcule un score composite de rentabilité & conviction [0..100] pour
        classer automatiquement les meilleures actions éligibles à l'achat.
        Ne modifie aucune stratégie : utilise uniquement la sortie du modèle
        entraîné et les déclencheurs multi-régimes existants.
        """
        m = metrics or {}
        t = triggers or {}

        # 1. Conviction IA (poids majeur : 60%)
        prob_component = max(0.0, min(1.0, float(prob_buy))) * 60.0

        # 2. Qualité historique d'apprentissage : Sharpe & Profit Factor (25%)
        sharpe = float(m.get("Sharpe Ratio", 0.0) or 0.0)
        pf_val = float(m.get("Profit Factor", 1.0) or 1.0)
        accuracy = float(m.get("accuracy", 0.5) or 0.5)
        sharpe_norm = max(0.0, min(1.0, (sharpe + 1.0) / 4.0))
        pf_norm = max(0.0, min(1.0, (pf_val - 0.5) / 2.5))
        acc_norm = max(0.0, min(1.0, (accuracy - 0.4) / 0.4))
        quality_component = (0.45 * sharpe_norm + 0.35 * pf_norm + 0.20 * acc_norm) * 25.0

        # 3. Bonus de régime institutionnel (15%)
        regime_bonus = 0.0
        if t.get("trend_regime", {}).get("status") == "BULLISH":
            regime_bonus += 6.0
        if t.get("volume_surge", {}).get("status") == "INSTITUTIONAL_INFLOW":
            regime_bonus += 5.0
        if t.get("rsi_shield", {}).get("status") == "OVERSOLD_OPPORTUNITY":
            regime_bonus += 4.0
        elif t.get("bollinger_squeeze", {}).get("status") == "SQUEEZE_ACTIVE":
            regime_bonus += 2.0

        return round(prob_component + quality_component + regime_bonus, 2)

    # -------------------------------------------------------------------------
    # Évaluation d'un ticker (données + features + prédiction + triggers)
    # -------------------------------------------------------------------------
    def _evaluate_ticker(
        self,
        symbol: str,
        macro_df: Optional[pd.DataFrame],
        current_vix: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        trader = self._get_trader(symbol)
        if trader is None:
            return None

        df_raw = self._data_fetcher(symbol, self.config.period, self.config.interval)
        if df_raw is None or df_raw.empty or len(df_raw) < 30:
            return None

        df = add_features(df_raw, labeling_mode=self.config.labeling_mode)
        if macro_df is not None:
            df = join_macro_data(df, macro_df)
        if df.empty:
            return None

        last_row = df.iloc[-1:]
        features = trader.get_feature_list(df)
        prob_buy = trader.predict(last_row, features)
        if prob_buy is None:
            return None

        current_price = float(last_row["Close"].iloc[0])
        if current_price <= 0:
            return None

        raw_atr = last_row["ATR"].iloc[0] if "ATR" in last_row.columns else np.nan
        current_atr = float(raw_atr) if pd.notna(raw_atr) and float(raw_atr) > 0 else (current_price * 0.02)

        triggers = evaluate_multi_regime_triggers(
            last_row.iloc[0],
            current_price=current_price,
            current_atr=current_atr,
            trader_prob=prob_buy,
            current_vix=current_vix,
        )

        with self._lock:
            metrics = dict(self._model_metrics.get(symbol, {}))

        score = self.compute_candidate_score(prob_buy, metrics, triggers)

        return {
            "symbol": symbol,
            "name": get_company_name_from_yahoo(symbol),
            "price": current_price,
            "atr": current_atr,
            "prob_buy": float(prob_buy),
            "score": score,
            "triggers": triggers,
            "metrics": metrics,
            "trend_status": triggers.get("trend_regime", {}).get("status", "UNKNOWN"),
            "rsi_status": triggers.get("rsi_shield", {}).get("status", "NORMAL"),
            "vix_status": triggers.get("macro_vix", {}).get("status", "CALM_MARKET"),
        }

    # -------------------------------------------------------------------------
    # Calcul de l'équité totale du portefeuille
    # -------------------------------------------------------------------------
    def _compute_portfolio_equity(self, pf: Dict[str, Any], price_cache: Dict[str, float]) -> float:
        cash = float(pf.get("cash", 0.0))
        pos_val = 0.0
        for sym, pos in pf.get("positions", {}).items():
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue
            px = price_cache.get(sym)
            if px is None:
                px = self._price_fetcher(sym) or float(pos.get("avg_price", 0.0))
                if px > 0:
                    price_cache[sym] = px
            pos_val += qty * float(px)
        return cash + pos_val

    # -------------------------------------------------------------------------
    # Exécution d'un cycle autonome complet (Vente -> Sélection -> Achat)
    # -------------------------------------------------------------------------
    def run_single_cycle(self) -> Dict[str, Any]:
        """
        Exécute une passe complète du cycle automatique :
        - Étape 1 : Évaluation des positions ouvertes (SL / TP / Trailing / Breakeven / Signal IA)
                    -> Vente automatique & libération immédiate du cash
        - Étape 2 : Vérification des coupe-circuits (VIX, Drawdown Max)
        - Étape 3 : Scan & classement des actions entraînées issues des CSV
                    -> Achat automatique des meilleures opportunités
        """
        cycle_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg = self.config
        pf = self._load_pf()
        if not isinstance(pf.get("positions"), dict):
            pf["positions"] = {}
        if not isinstance(pf.get("history"), list):
            pf["history"] = []

        macro_df = self._macro_fetcher(cfg.period, cfg.interval)
        current_vix: Optional[float] = None
        if macro_df is not None and not macro_df.empty and "VIX" in macro_df.columns:
            vix_series = macro_df["VIX"].dropna()
            if not vix_series.empty:
                current_vix = float(vix_series.iloc[-1])

        price_cache: Dict[str, float] = {}
        eval_cache: Dict[str, Dict[str, Any]] = {}
        sells_executed: List[Dict[str, Any]] = []
        buys_executed: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # PHASE 1 : SURVEILLANCE & VENTE AUTOMATIQUE DES POSITIONS OUVERTES
        # ---------------------------------------------------------------------
        for sym, pos in list(pf.get("positions", {}).items()):
            if self._stop_event.is_set():
                break
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue

            avg_price = float(pos.get("avg_price", 0.0))
            eval_res = self._evaluate_ticker(sym, macro_df, current_vix)
            live_px = self._price_fetcher(sym) if cfg.use_live_intraday_price else None
            if eval_res is not None:
                eval_cache[sym] = eval_res
                current_price = float(live_px) if (live_px is not None and float(live_px) > 0) else eval_res["price"]
                current_atr = eval_res["atr"]
                prob_buy = eval_res["prob_buy"]
            else:
                fetched_px = live_px if (live_px is not None and float(live_px) > 0) else self._price_fetcher(sym)
                if fetched_px is None or fetched_px <= 0:
                    continue
                current_price = float(fetched_px)
                current_atr = current_price * 0.02
                prob_buy = 0.50

            price_cache[sym] = current_price
            pnl_pct = ((current_price - avg_price) / avg_price * 100.0) if avg_price > 0 else 0.0

            sl_price = pos.get("stop_loss")
            tp_price = pos.get("take_profit")

            # Suivi du plus haut prix atteint depuis l'ouverture (base du Trailing Stop serré)
            prev_high = float(pos.get("highest_price") or avg_price or current_price)
            highest_price = max(prev_high, current_price)
            if highest_price > prev_high or "highest_price" not in pos:
                pos["highest_price"] = highest_price
                self._save_pf(pf)

            # 1a. Breakeven Stop automatique à +1R (verrouillage 0 risque)
            if cfg.use_breakeven_stop and sl_price is not None and avg_price > 0 and sl_price < avg_price:
                initial_risk = max(avg_price - sl_price, avg_price * 0.01)
                if current_price >= avg_price + initial_risk:
                    pos["stop_loss"] = avg_price
                    sl_price = avg_price
                    self._save_pf(pf)
                    self._log("INFO", f"Breakeven Stop activé : SL remonté à {avg_price:.2f} $", ticker=sym)

            # 1b. Trailing Stop serré dynamique (ATR + % depuis le plus haut, ne redescend jamais)
            if cfg.use_trailing_stop and current_price > avg_price:
                tsl_candidates: List[float] = []
                if current_atr > 0 and cfg.trailing_atr_mult > 0:
                    tsl_candidates.append(highest_price - (cfg.trailing_atr_mult * current_atr))
                if cfg.trailing_stop_pct > 0:
                    tsl_candidates.append(highest_price * (1.0 - cfg.trailing_stop_pct / 100.0))
                if tsl_candidates:
                    candidate_tsl = max(tsl_candidates)
                    if sl_price is None or candidate_tsl > sl_price:
                        pos["stop_loss"] = candidate_tsl
                        sl_price = candidate_tsl
                        self._save_pf(pf)
                        self._log(
                            "INFO",
                            f"Trailing Stop remonté à {sl_price:.2f} $ (plus haut: {highest_price:.2f} $)",
                            ticker=sym,
                        )

            # 1c. Décision de vente totale selon critères de rentabilité & risque
            sell_reason: Optional[str] = None
            if cfg.take_profit_pct > 0 and pnl_pct >= cfg.take_profit_pct:
                sell_reason = f"🎯 Take-Profit Rentabilité (+{pnl_pct:.2f}% >= +{cfg.take_profit_pct:.1f}%)"
            elif cfg.use_atr_stops and tp_price is not None and current_price >= tp_price:
                sell_reason = f"🎯 Take-Profit ATR atteint ({current_price:.2f} >= {tp_price:.2f})"
            elif cfg.stop_loss_pct > 0 and pnl_pct <= -cfg.stop_loss_pct:
                sell_reason = f"🛑 Stop-Loss Max Perte ({pnl_pct:.2f}% <= -{cfg.stop_loss_pct:.1f}%)"
            elif (cfg.use_atr_stops or cfg.use_trailing_stop) and sl_price is not None and current_price <= sl_price:
                sell_reason = f"🛑 Stop-Loss / Trailing atteint ({current_price:.2f} <= {sl_price:.2f})"
            elif eval_res is not None and prob_buy <= cfg.sell_prob_threshold:
                sell_reason = f"🔴 Signal IA Baissier (Prob={prob_buy*100:.1f}% <= {cfg.sell_prob_threshold*100:.0f}%)"

            if sell_reason:
                est_pnl = (current_price - avg_price) * qty
                if cfg.dry_run:
                    self._log("SELL", f"[DRY-RUN] Vente {qty:.2f} {sym} @ {current_price:.2f} $ — {sell_reason}", ticker=sym)
                    self._record_trade_event("SELL", sym, qty, current_price, sell_reason, pnl=est_pnl, dry_run=True)
                    sells_executed.append({"symbol": sym, "qty": qty, "price": current_price, "pnl": est_pnl, "reason": sell_reason})
                else:
                    ok, msg = execute_equity_trade(pf, sym, qty, "SELL", current_price)
                    if ok:
                        self._save_pf(pf)
                        realized = pf["history"][-1].get("pnl", est_pnl) if pf["history"] else est_pnl
                        with self._lock:
                            self._total_sells += 1
                            self._realized_pnl += float(realized)
                            self._last_sell_ts[sym] = time.time()
                        self._log("SELL", f"Vente auto {qty:.2f} {sym} @ {current_price:.2f} $ (PnL: {realized:+.2f} $) — {sell_reason}", ticker=sym)
                        self._record_trade_event("SELL", sym, qty, current_price, sell_reason, pnl=float(realized), dry_run=False)
                        sells_executed.append({"symbol": sym, "qty": qty, "price": current_price, "pnl": float(realized), "reason": sell_reason})
                    else:
                        self._log("WARN", f"Échec vente {sym}: {msg}", ticker=sym)
            # 1d. Prise de profit partielle automatique (sécurise une partie des gains + remonte SL au Breakeven)
            elif (
                cfg.use_partial_take_profit
                and cfg.partial_take_profit_pct > 0
                and not pos.get("partial_tp_taken", False)
                and pnl_pct >= cfg.partial_take_profit_pct
                and qty >= 2
            ):
                ratio = max(0.1, min(0.9, float(cfg.partial_take_profit_ratio)))
                partial_qty = max(1, min(int(qty) - 1, int(round(qty * ratio))))
                partial_reason = (
                    f"💰 Prise de Profit Partielle ({int(ratio*100)}%) "
                    f"(+{pnl_pct:.2f}% >= +{cfg.partial_take_profit_pct:.1f}%)"
                )
                est_partial_pnl = (current_price - avg_price) * partial_qty
                if cfg.dry_run:
                    self._log(
                        "SELL",
                        f"[DRY-RUN] Vente partielle {partial_qty}/{int(qty)} {sym} @ {current_price:.2f} $ — {partial_reason}",
                        ticker=sym,
                    )
                    self._record_trade_event("SELL (PARTIAL)", sym, partial_qty, current_price, partial_reason, pnl=est_partial_pnl, dry_run=True)
                    sells_executed.append({"symbol": sym, "qty": partial_qty, "price": current_price, "pnl": est_partial_pnl, "reason": partial_reason})
                else:
                    ok, msg = execute_equity_trade(pf, sym, partial_qty, "SELL (Partial TP)", current_price)
                    if ok:
                        rem_pos = pf["positions"].get(sym)
                        if rem_pos is not None:
                            rem_pos["partial_tp_taken"] = True
                            if avg_price > 0 and (rem_pos.get("stop_loss") is None or rem_pos["stop_loss"] < avg_price):
                                rem_pos["stop_loss"] = avg_price
                        self._save_pf(pf)
                        realized = pf["history"][-1].get("pnl", est_partial_pnl) if pf["history"] else est_partial_pnl
                        with self._lock:
                            self._total_sells += 1
                            self._realized_pnl += float(realized)
                        self._log(
                            "SELL",
                            f"Vente partielle {partial_qty} {sym} @ {current_price:.2f} $ (PnL: {realized:+.2f} $) — {partial_reason} | SL sécurisé au PRU ({avg_price:.2f} $)",
                            ticker=sym,
                        )
                        self._record_trade_event("SELL (PARTIAL)", sym, partial_qty, current_price, partial_reason, pnl=float(realized), dry_run=False)
                        sells_executed.append({"symbol": sym, "qty": partial_qty, "price": current_price, "pnl": float(realized), "reason": partial_reason})
                    else:
                        self._log("WARN", f"Échec vente partielle {sym}: {msg}", ticker=sym)

        # ---------------------------------------------------------------------
        # PHASE 2 : CALCUL ÉQUITÉ & COUPE-CIRCUITS INSTITUTIONNELS
        # ---------------------------------------------------------------------
        equity = self._compute_portfolio_equity(pf, price_cache)
        with self._lock:
            if equity > self._peak_equity:
                self._peak_equity = equity
            self._current_equity = equity
            dd_pct = ((self._peak_equity - equity) / self._peak_equity * 100.0) if self._peak_equity > 0 else 0.0
            self._drawdown_pct = dd_pct
            self._last_vix = current_vix
            self._drawdown_halted = bool(cfg.max_drawdown_pct > 0 and dd_pct >= cfg.max_drawdown_pct)

        vix_halted = bool(cfg.use_vix_breaker and current_vix is not None and current_vix > cfg.vix_max_threshold)

        # Point 2 : Régime de marché global (SPY > SMA200 & VIX < 20 -> 100%, sinon taille défensive)
        spy_above_sma200 = True
        if macro_df is not None and not macro_df.empty and "SPY_Above_SMA200" in macro_df.columns:
            spy_s = macro_df["SPY_Above_SMA200"].dropna()
            if not spy_s.empty:
                spy_above_sma200 = bool(float(spy_s.iloc[-1]) >= 0.5)

        is_defensive_regime = (
            (current_vix is not None and current_vix >= 20.0)
            or (not spy_above_sma200)
        )
        if cfg.use_adaptive_regime_sizing and is_defensive_regime:
            regime_size_mult = max(0.1, min(1.0, float(cfg.defensive_size_multiplier)))
            reasons_def = []
            if current_vix is not None and current_vix >= 20.0:
                reasons_def.append(f"VIX={current_vix:.1f}")
            if not spy_above_sma200:
                reasons_def.append("SPY<SMA200")
            market_regime_lbl = f"DÉFENSIF ({', '.join(reasons_def)} • x{regime_size_mult:.2f})"
        else:
            regime_size_mult = 1.0
            market_regime_lbl = "HAUSSIER CALME (x1.00)"

        effective_pos_pct = cfg.position_size_pct * regime_size_mult
        with self._lock:
            self._market_regime = market_regime_lbl
            self._effective_position_pct = round(effective_pos_pct, 2)

        # ---------------------------------------------------------------------
        # PHASE 3 : SCAN DES TRACKERS & SÉLECTION AUTOMATIQUE D'ACHAT
        # ---------------------------------------------------------------------
        candidates: List[Dict[str, Any]] = []
        all_rankings: List[Dict[str, Any]] = []

        for raw_sym in cfg.tickers:
            if self._stop_event.is_set():
                break
            sym = convert_google_to_yahoo_ticker(raw_sym)
            if sym in eval_cache:
                ev = eval_cache[sym]
            else:
                ev = self._evaluate_ticker(sym, macro_df, current_vix)
                if ev is not None:
                    eval_cache[sym] = ev

            if ev is None:
                continue

            price_cache[sym] = ev["price"]
            already_held = sym in pf.get("positions", {}) and pf["positions"][sym].get("qty", 0) > 0
            last_sell = self._last_sell_ts.get(sym, 0.0)
            in_cooldown = (time.time() - last_sell) < cfg.cooldown_sec

            status_note = "ÉLIGIBLE"
            eligible = True

            wfa_pf = ev["metrics"].get("Profit Factor") if isinstance(ev.get("metrics"), dict) else None
            wfa_sharpe = ev["metrics"].get("Sharpe Ratio") if isinstance(ev.get("metrics"), dict) else None

            if already_held:
                eligible = False
                status_note = "EN PORTEFEUILLE"
            elif in_cooldown:
                eligible = False
                status_note = "COOLDOWN"
            elif vix_halted:
                eligible = False
                status_note = f"BLOQUÉ VIX ({current_vix:.1f})"
            elif self._drawdown_halted:
                eligible = False
                status_note = f"BLOQUÉ DRAWDOWN ({dd_pct:.1f}%)"
            elif ev["prob_buy"] < cfg.min_buy_prob:
                eligible = False
                status_note = f"CONVICTION < {cfg.min_buy_prob*100:.0f}%"
            elif cfg.use_wfa_quality_filter and wfa_pf is not None and float(wfa_pf) < cfg.min_wfa_profit_factor:
                eligible = False
                status_note = f"QUALITÉ WFA (PF {float(wfa_pf):.2f} < {cfg.min_wfa_profit_factor:.2f})"
            elif cfg.use_wfa_quality_filter and wfa_sharpe is not None and float(wfa_sharpe) < cfg.min_wfa_sharpe:
                eligible = False
                status_note = f"QUALITÉ WFA (Sharpe {float(wfa_sharpe):.2f} < {cfg.min_wfa_sharpe:.2f})"
            elif cfg.use_rsi_shield and ev["rsi_status"] == "OVERBOUGHT_SHIELD":
                eligible = False
                status_note = "BOUCLIER RSI (>75)"
            elif cfg.use_sma200_filter and ev["trend_status"] == "BEARISH":
                eligible = False
                status_note = "SOUS SMA200"

            rank_row = {
                "symbol": sym,
                "name": ev["name"],
                "price": round(ev["price"], 2),
                "prob_buy_pct": round(ev["prob_buy"] * 100.0, 1),
                "score": ev["score"],
                "wfa_pf": round(float(wfa_pf), 2) if wfa_pf is not None else None,
                "wfa_sharpe": round(float(wfa_sharpe), 2) if wfa_sharpe is not None else None,
                "trend": ev["trend_status"],
                "rsi": ev["rsi_status"],
                "status": status_note,
            }
            all_rankings.append(rank_row)

            if eligible:
                candidates.append(ev)

        # Tri par meilleur score composite de rentabilité & conviction
        candidates.sort(key=lambda x: (x["score"], x["prob_buy"]), reverse=True)
        all_rankings.sort(key=lambda x: (x["score"], x["prob_buy_pct"]), reverse=True)

        # Exécution des achats jusqu'à max_open_positions en respectant la réserve de cash
        open_count = sum(1 for p in pf.get("positions", {}).values() if p.get("qty", 0) > 0)
        slots_available = max(0, cfg.max_open_positions - open_count)
        min_cash_reserve = equity * (cfg.min_cash_reserve_pct / 100.0)

        for cand in candidates:
            if self._stop_event.is_set() or slots_available <= 0:
                break

            avail_cash = float(pf.get("cash", 0.0)) - min_cash_reserve
            if avail_cash <= 0:
                break

            sym = cand["symbol"]
            target_alloc = min(equity * (effective_pos_pct / 100.0), avail_cash)
            live_buy_px = self._price_fetcher(sym) if cfg.use_live_intraday_price else None
            px = float(live_buy_px) if (live_buy_px is not None and float(live_buy_px) > 0) else cand["price"]
            # Prévoir marge slippage + commission (0.2%)
            qty = int(target_alloc / (px * 1.002)) if px > 0 else 0
            if qty <= 0:
                continue

            atr_val = cand["atr"]
            buy_reason = (
                f"🟢 Top Score {cand['score']:.1f}/100 | Prob IA {cand['prob_buy']*100:.1f}% | "
                f"Alloc {effective_pos_pct:.1f}% ({market_regime_lbl})"
            )

            if cfg.dry_run:
                self._log("BUY", f"[DRY-RUN] Achat {qty} {sym} @ {px:.2f} $ — {buy_reason}", ticker=sym)
                self._record_trade_event("BUY", sym, qty, px, buy_reason, dry_run=True)
                buys_executed.append({"symbol": sym, "qty": qty, "price": px, "score": cand["score"], "reason": buy_reason})
                slots_available -= 1
            else:
                ok, msg = execute_equity_trade(pf, sym, qty, "BUY", px, current_atr=atr_val)
                if ok:
                    # Ajuster SL/TP si l'utilisateur a défini des seuils % plus protecteurs/ciblés
                    pos = pf["positions"].get(sym)
                    if pos is not None:
                        exec_px = pos["avg_price"]
                        if not cfg.use_atr_stops:
                            if cfg.stop_loss_pct > 0:
                                pos["stop_loss"] = exec_px * (1.0 - cfg.stop_loss_pct / 100.0)
                            if cfg.take_profit_pct > 0:
                                pos["take_profit"] = exec_px * (1.0 + cfg.take_profit_pct / 100.0)
                    self._save_pf(pf)
                    with self._lock:
                        self._total_buys += 1
                    self._log("BUY", f"Achat auto {qty} {sym} @ {px:.2f} $ — {buy_reason}", ticker=sym)
                    self._record_trade_event("BUY", sym, qty, px, buy_reason, dry_run=False)
                    buys_executed.append({"symbol": sym, "qty": qty, "price": px, "score": cand["score"], "reason": buy_reason})
                    slots_available -= 1
                else:
                    self._log("WARN", f"Échec achat {sym}: {msg}", ticker=sym)

        # Mise à jour finale des métriques de cycle
        equity_after = self._compute_portfolio_equity(pf, price_cache)
        with self._lock:
            self._cycle_count += 1
            self._last_cycle_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._current_equity = equity_after
            if equity_after > self._peak_equity:
                self._peak_equity = equity_after
            self._last_rankings = all_rankings[:50]
            self._equity_history.append({
                "time": self._last_cycle_at,
                "cycle": self._cycle_count,
                "equity": round(equity_after, 2),
                "cash": round(float(pf.get("cash", 0.0)), 2),
                "open_positions": sum(1 for p in pf.get("positions", {}).values() if p.get("qty", 0) > 0),
            })
            if len(self._equity_history) > 500:
                self._equity_history = self._equity_history[-500:]

        self._log(
            "CYCLE",
            f"Cycle #{self._cycle_count} terminé : {len(sells_executed)} vente(s), "
            f"{len(buys_executed)} achat(s) | Équité : {equity_after:,.2f} $ | Cash : {pf.get('cash', 0.0):,.2f} $",
        )
        self._persist_state()

        return {
            "cycle": self._cycle_count,
            "started_at": cycle_start,
            "finished_at": self._last_cycle_at,
            "sells": sells_executed,
            "buys": buys_executed,
            "equity": equity_after,
            "cash": float(pf.get("cash", 0.0)),
            "open_positions": len(pf.get("positions", {})),
            "rankings_count": len(all_rankings),
        }

    # -------------------------------------------------------------------------
    # Boucle d'arrière-plan (Start / Stop / Pause / Resume / Emergency Liquidate)
    # -------------------------------------------------------------------------
    def _loop(self) -> None:
        self._log("SYSTEM", f"🚀 Mode Automatique démarré sur {len(self.config.tickers)} ticker(s) (intervalle: {self.config.scan_interval_sec}s).")
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._stop_event.wait(timeout=1.0)
                continue
            try:
                self.run_single_cycle()
                with self._lock:
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                self._log("ERROR", f"Erreur durant le cycle automatique : {exc}")
                self._persist_state()

            # Attente interruptible instantanément par stop()
            interval = max(1, int(self.config.scan_interval_sec))
            self._stop_event.wait(timeout=interval)

        with self._lock:
            self._is_running = False
            self._stopped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log("SYSTEM", "🛑 Mode Automatique interrompu proprement.")
        self._persist_state()

    def start(
        self,
        config: AutoTraderConfig,
        trained_models: Optional[Dict[str, MLTrader]] = None,
        model_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        """Démarre la boucle de trading automatique en arrière-plan."""
        with self._lock:
            if self._is_running and self._thread is not None and self._thread.is_alive():
                return False, "Le mode automatique est déjà en cours d'exécution."

            if not config.tickers:
                return False, "Aucun ticker configuré pour le mode automatique."

            self.config = config
            if trained_models:
                self.register_models(trained_models, model_metrics)

            # Vérifier qu'au moins un modèle entraîné est disponible
            ready_count = sum(1 for t in config.tickers if self._get_trader(convert_google_to_yahoo_ticker(t)) is not None)
            if ready_count == 0:
                return False, "Aucun modèle entraîné disponible. Lancez d'abord l'apprentissage (WFA ou Standard)."

            self._stop_event.clear()
            self._pause_event.clear()
            self._is_running = True
            self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._stopped_at = None
            self._last_error = None

            self._thread = threading.Thread(target=self._loop, name="AutoTraderLoop", daemon=True)
            self._thread.start()
            return True, f"Mode automatique démarré ({ready_count}/{len(config.tickers)} modèles prêts)."

    def stop(self, timeout: float = 5.0) -> Tuple[bool, str]:
        """Interrompt immédiatement la boucle de trading automatique."""
        self._stop_event.set()
        self._pause_event.clear()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=timeout)
        with self._lock:
            self._is_running = False
            self._stopped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._persist_state()
        return True, "Mode automatique arrêté."

    def pause(self) -> None:
        """Met la boucle en pause sans perdre l'état."""
        self._pause_event.set()
        self._log("SYSTEM", "⏸️ Mode Automatique mis en pause.")
        self._persist_state()

    def resume(self) -> None:
        """Reprend la boucle après une pause."""
        self._pause_event.clear()
        self._log("SYSTEM", "▶️ Mode Automatique repris.")
        self._persist_state()

    def update_config(self, config: AutoTraderConfig) -> None:
        """Met à jour la configuration du moteur à chaud sans interrompre la boucle."""
        with self._lock:
            self.config = config
        self._log(
            "SYSTEM",
            f"⚙️ Configuration mise à jour à chaud (Max pos={config.max_open_positions}, "
            f"TP={config.take_profit_pct:.1f}%, SL={config.stop_loss_pct:.1f}%, "
            f"Vente IA<={config.sell_prob_threshold*100:.0f}%).",
        )
        self._persist_state()

    def emergency_liquidate_all(self) -> Tuple[int, float, str]:
        """
        Commande Expert (Kill Switch) :
        1. Interrompt immédiatement le mode automatique
        2. Vend toutes les positions ouvertes au prix du marché et repasse 100% Cash.
        """
        self.stop(timeout=3.0)
        pf = self._load_pf()
        positions = dict(pf.get("positions", {}))
        if not positions:
            self._log("SYSTEM", "🚨 Liquidation d'urgence demandée : aucune position ouverte.")
            return 0, 0.0, "Aucune position ouverte à liquider."

        closed_count = 0
        total_pnl = 0.0
        for sym, pos in positions.items():
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue
            px = self._price_fetcher(sym) or float(pos.get("avg_price", 0.0))
            if px <= 0:
                continue
            ok, _ = execute_equity_trade(pf, sym, qty, "SELL (Emergency Kill Switch)", px)
            if ok:
                pnl = float(pf["history"][-1].get("pnl", 0.0)) if pf["history"] else 0.0
                total_pnl += pnl
                closed_count += 1
                self._record_trade_event("SELL (KILL-SWITCH)", sym, qty, px, "🚨 Liquidation totale d'urgence", pnl=pnl)

        if closed_count > 0:
            self._save_pf(pf)
            with self._lock:
                self._total_sells += closed_count
                self._realized_pnl += total_pnl

        msg = f"🚨 Liquidation d'urgence terminée : {closed_count} position(s) clôturée(s), PnL réalisé : {total_pnl:+.2f} $."
        self._log("SYSTEM", msg)
        self._persist_state()
        return closed_count, total_pnl, msg

    def get_status(self) -> Dict[str, Any]:
        """Retourne un instantané thread-safe de l'état complet du moteur."""
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {
                "is_running": bool(self._is_running and alive),
                "is_paused": bool(self._pause_event.is_set()),
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "last_cycle_at": self._last_cycle_at,
                "last_error": self._last_error,
                "cycle_count": self._cycle_count,
                "total_buys": self._total_buys,
                "total_sells": self._total_sells,
                "realized_pnl": round(self._realized_pnl, 2),
                "peak_equity": round(self._peak_equity, 2),
                "current_equity": round(self._current_equity, 2),
                "drawdown_pct": round(self._drawdown_pct, 2),
                "drawdown_halted": self._drawdown_halted,
                "last_vix": self._last_vix,
                "market_regime": self._market_regime,
                "effective_position_pct": self._effective_position_pct,
                "trained_models_count": len(self._trained_models),
                "trained_symbols": sorted(self._trained_models.keys()),
                "last_rankings": list(self._last_rankings),
                "logs": list(self._logs),
                "trade_events": list(self._trade_events),
                "equity_history": list(self._equity_history),
                "config": self.config.to_dict(),
            }


# Singleton global partagé par l'application web
_GLOBAL_ENGINE: Optional[AutoTraderEngine] = None
_GLOBAL_ENGINE_LOCK = threading.Lock()


def get_global_auto_trader() -> AutoTraderEngine:
    """Retourne l'instance singleton du moteur de trading automatique."""
    global _GLOBAL_ENGINE
    with _GLOBAL_ENGINE_LOCK:
        if _GLOBAL_ENGINE is None:
            _GLOBAL_ENGINE = AutoTraderEngine()
        return _GLOBAL_ENGINE
