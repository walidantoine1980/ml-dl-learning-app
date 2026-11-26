"""
===============================================================================
SUITE DE TESTS AUTOMATISÉS : MODE TRADING AUTOMATIQUE (AUTOPILOT) & MULTI-CSV
XGBoost Stock Predictor Pro - Test Suite
===============================================================================
Exécution :
    python tests/test_auto_trader.py
    ou
    pytest tests/
===============================================================================
"""

import os
import sys
import tempfile
import time
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from auto_trader_engine import (
    AutoTraderConfig,
    AutoTraderEngine,
    list_watchlist_csv_files,
    load_auto_trader_config,
    load_tickers_from_csv_files,
    save_auto_trader_config,
)
from quant_core import load_portfolio, save_portfolio


class _StubTrader:
    """Stub léger imitant un MLTrader déjà entraîné (pour tests déterministes hors-ligne)."""

    def __init__(self, prob_value: float = 0.75, accuracy: float = 0.68, sharpe: float = 1.6, pf: float = 1.9):
        self.is_trained = True
        self.prob_value = prob_value
        self.accuracy = accuracy
        self.advanced_metrics = {
            "Sharpe Ratio": sharpe,
            "Profit Factor": pf,
            "VaR (95%)": -0.02,
        }

    def get_feature_list(self, df):
        return ["Close"]

    def predict(self, last_row, features):
        return self.prob_value


def _make_ohlcv_df(base_price: float = 100.0, n_days: int = 80, final_price: float | None = None) -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique valide pour add_features()."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0005, 0.008, size=n_days)
    prices = base_price * np.exp(np.cumsum(noise))
    if final_price is not None:
        prices[-1] = final_price
    high = prices * 1.01
    low = prices * 0.99
    open_p = prices * 1.00
    volume = np.full(n_days, 1_000_000, dtype=int)
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": prices, "Volume": volume},
        index=dates,
    )


class TestAutoTraderEngine(unittest.TestCase):
    """Tests unitaires et d'intégration pour le mode automatique et la sélection multi-CSV."""

    def test_01_multi_csv_discovery_and_merge(self):
        """Vérifie la découverte des CSV dans WatchList/ et la fusion dédupliquée multi-fichiers."""
        real_csvs = list_watchlist_csv_files()
        self.assertGreater(len(real_csvs), 0, "Le dossier WatchList/ doit contenir des fichiers CSV.")

        # Test avec 2 fichiers CSV temporaires contenant des doublons et suffixes Reuters/Google
        with tempfile.TemporaryDirectory() as tmpdir:
            csv1 = os.path.join(tmpdir, "tech_Watchlist.csv")
            csv2 = os.path.join(tmpdir, "eu_Watchlist.csv")
            with open(csv1, "w", encoding="utf-8") as f:
                f.write("Name;Symbol;Exchange\nApple;AAPL;NASDAQ\nOracle;ORCL.K;NYSE\n")
            with open(csv2, "w", encoding="utf-8") as f:
                f.write("Name,Symbol,Exchange\nApple Inc,AAPL,NASDAQ\nLVMH,EPA:MC,EPA\n")

            files = list_watchlist_csv_files(tmpdir)
            self.assertEqual(len(files), 2)

            merged = load_tickers_from_csv_files(files, watchlists_dir=tmpdir)
            self.assertEqual(len(merged["errors"]), 0)
            self.assertEqual(merged["total_unique"], 3)
            self.assertEqual(set(merged["symbols"]), {"AAPL", "ORCL", "MC.PA"})

    def test_02_config_persistence_roundtrip(self):
        """Vérifie la sauvegarde et le rechargement atomique de AutoTraderConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "auto_cfg.json")
            cfg = AutoTraderConfig(
                tickers=["AAPL", "MSFT", "MC.PA"],
                csv_sources=["a.csv", "b.csv"],
                scan_interval_sec=30,
                max_open_positions=4,
                take_profit_pct=7.5,
                stop_loss_pct=2.5,
                learning_mode="Walk-Forward Analysis (WFA)",
            )
            save_auto_trader_config(cfg, filepath=cfg_path)
            reloaded = load_auto_trader_config(filepath=cfg_path)
            self.assertEqual(reloaded.tickers, ["AAPL", "MSFT", "MC.PA"])
            self.assertEqual(reloaded.csv_sources, ["a.csv", "b.csv"])
            self.assertEqual(reloaded.take_profit_pct, 7.5)
            self.assertEqual(reloaded.learning_mode, "Walk-Forward Analysis (WFA)")

    def test_03_candidate_scoring_ranking(self):
        """Vérifie que le score composite privilégie la meilleure rentabilité/conviction."""
        strong = AutoTraderEngine.compute_candidate_score(
            prob_buy=0.80,
            metrics={"Sharpe Ratio": 2.0, "Profit Factor": 2.4, "accuracy": 0.72},
            triggers={"trend_regime": {"status": "BULLISH"}, "volume_surge": {"status": "INSTITUTIONAL_INFLOW"}},
        )
        weak = AutoTraderEngine.compute_candidate_score(
            prob_buy=0.55,
            metrics={"Sharpe Ratio": 0.2, "Profit Factor": 1.0, "accuracy": 0.51},
            triggers={"trend_regime": {"status": "BEARISH"}},
        )
        self.assertGreater(strong, weak)

    def test_04_full_autonomous_buy_sell_rebuy_cycle(self):
        """Vérifie la boucle complète : Achat auto -> Take-Profit Vente auto -> Ré-achat auto."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "state.json")
            save_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=pf_path, allow_reset=True)

            # Prix pilotés par étape
            price_state = {"AAPL": 100.0, "MSFT": 200.0}
            stubs = {
                "AAPL": _StubTrader(prob_value=0.82, sharpe=2.1, pf=2.3),
                "MSFT": _StubTrader(prob_value=0.74, sharpe=1.5, pf=1.8),
            }

            def mock_fetch(sym, period, interval):
                return _make_ohlcv_df(base_price=price_state[sym], final_price=price_state[sym])

            def mock_macro(period, interval):
                dates = pd.date_range("2025-01-01", periods=80, freq="B")
                return pd.DataFrame({"VIX": np.full(80, 16.0)}, index=dates)

            engine = AutoTraderEngine(
                data_fetcher=mock_fetch,
                macro_fetcher=mock_macro,
                price_fetcher=lambda s: price_state[s],
                state_file=state_path,
            )
            cfg = AutoTraderConfig(
                tickers=["AAPL", "MSFT"],
                max_open_positions=1,
                position_size_pct=20.0,
                min_cash_reserve_pct=10.0,
                min_buy_prob=0.60,
                sell_prob_threshold=0.45,
                take_profit_pct=5.0,
                stop_loss_pct=3.0,
                use_rsi_shield=False,
                cooldown_sec=0,
                portfolio_path=pf_path,
            )
            engine.config = cfg
            engine.register_models(stubs)

            # CYCLE 1 : doit sélectionner et acheter AAPL (score le plus élevé)
            r1 = engine.run_single_cycle()
            self.assertEqual(len(r1["buys"]), 1)
            self.assertEqual(r1["buys"][0]["symbol"], "AAPL")
            pf1 = load_portfolio(filepath=pf_path)
            self.assertIn("AAPL", pf1["positions"])

            # CYCLE 2 : AAPL monte de +10% (>= TP 5%) -> Vente automatique + Ré-achat dans la foulée
            price_state["AAPL"] = 110.0
            stubs["AAPL"].prob_value = 0.50  # Neutre après prise de profit
            r2 = engine.run_single_cycle()
            self.assertEqual(len(r2["sells"]), 1)
            self.assertEqual(r2["sells"][0]["symbol"], "AAPL")
            self.assertGreater(r2["sells"][0]["pnl"], 0.0, "Le Take-Profit doit générer un PnL positif.")
            # Cash libéré -> achat automatique de MSFT au même cycle !
            self.assertEqual(len(r2["buys"]), 1)
            self.assertEqual(r2["buys"][0]["symbol"], "MSFT")
            pf2 = load_portfolio(filepath=pf_path)
            self.assertNotIn("AAPL", pf2["positions"])
            self.assertIn("MSFT", pf2["positions"])

    def test_05_stop_loss_and_vix_circuit_breaker(self):
        """Vérifie la vente automatique sur Stop-Loss et le blocage d'achat si VIX > seuil."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "state.json")
            # Portefeuille avec position ouverte sur NVDA à 100 $
            save_portfolio(
                {
                    "cash": 80000.0,
                    "positions": {"NVDA": {"qty": 100, "avg_price": 100.0, "stop_loss": 95.0, "take_profit": 115.0}},
                    "history": [],
                },
                filepath=pf_path,
                allow_reset=True,
            )

            stubs = {
                "NVDA": _StubTrader(prob_value=0.70),
                "MSFT": _StubTrader(prob_value=0.85),
            }

            def mock_fetch(sym, period, interval):
                px = 90.0 if sym == "NVDA" else 200.0  # NVDA chute -10% -> Stop-Loss !
                return _make_ohlcv_df(base_price=100.0 if sym == "NVDA" else 200.0, final_price=px)

            def mock_macro_crisis(period, interval):
                dates = pd.date_range("2025-01-01", periods=80, freq="B")
                return pd.DataFrame({"VIX": np.full(80, 38.0)}, index=dates)  # Panique VIX > 30

            engine = AutoTraderEngine(
                data_fetcher=mock_fetch,
                macro_fetcher=mock_macro_crisis,
                price_fetcher=lambda s: 90.0 if s == "NVDA" else 200.0,
                state_file=state_path,
            )
            engine.config = AutoTraderConfig(
                tickers=["NVDA", "MSFT"],
                stop_loss_pct=5.0,
                use_vix_breaker=True,
                vix_max_threshold=30.0,
                use_rsi_shield=False,
                portfolio_path=pf_path,
            )
            engine.register_models(stubs)

            rep = engine.run_single_cycle()
            self.assertEqual(len(rep["sells"]), 1, "NVDA doit être vendue automatiquement par Stop-Loss.")
            self.assertEqual(len(rep["buys"]), 0, "Aucun achat ne doit avoir lieu quand le VIX dépasse 30.")

    def test_06_background_loop_start_pause_resume_and_clean_stop(self):
        """Vérifie le démarrage en boucle, la pause/reprise et l'interruption propre par stop()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "state.json")
            save_portfolio({"cash": 50000.0, "positions": {}, "history": []}, filepath=pf_path, allow_reset=True)

            engine = AutoTraderEngine(
                data_fetcher=lambda s, p, i: _make_ohlcv_df(100.0, final_price=100.0),
                macro_fetcher=lambda p, i: None,
                price_fetcher=lambda s: 100.0,
                state_file=state_path,
            )
            cfg = AutoTraderConfig(
                tickers=["AAPL"],
                scan_interval_sec=1,
                min_buy_prob=0.90,  # Pas d'ordre pour tester uniquement la boucle
                dry_run=True,
                portfolio_path=pf_path,
            )

            ok, msg = engine.start(cfg, trained_models={"AAPL": _StubTrader(prob_value=0.60)})
            self.assertTrue(ok, msg)
            time.sleep(1.3)

            st_running = engine.get_status()
            self.assertTrue(st_running["is_running"])
            self.assertGreaterEqual(st_running["cycle_count"], 1)

            engine.pause()
            self.assertTrue(engine.get_status()["is_paused"])
            engine.resume()
            self.assertFalse(engine.get_status()["is_paused"])

            ok_stop, _ = engine.stop(timeout=3.0)
            self.assertTrue(ok_stop)
            self.assertFalse(engine.get_status()["is_running"])

    def test_07_emergency_kill_switch_liquidate_all(self):
        """Vérifie que le Kill Switch interrompt le moteur et liquide 100% des positions en Cash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "state.json")
            save_portfolio(
                {
                    "cash": 50000.0,
                    "positions": {
                        "AAPL": {"qty": 10, "avg_price": 150.0},
                        "MSFT": {"qty": 5, "avg_price": 300.0},
                    },
                    "history": [],
                },
                filepath=pf_path,
                allow_reset=True,
            )

            engine = AutoTraderEngine(
                price_fetcher=lambda s: 160.0 if s == "AAPL" else 310.0,
                state_file=state_path,
            )
            engine.config = AutoTraderConfig(tickers=["AAPL", "MSFT"], portfolio_path=pf_path)

            closed, pnl, _ = engine.emergency_liquidate_all()
            self.assertEqual(closed, 2)
            self.assertGreater(pnl, 0.0)
            pf_after = load_portfolio(filepath=pf_path)
            self.assertEqual(len(pf_after["positions"]), 0)
            self.assertGreater(pf_after["cash"], 50000.0)

    def test_08_new_quant_features_and_time_decay_weights(self):
        """Vérifie les nouvelles features quantitatives (DI_Diff, ER_10, Donchian, MACD_Hist, etc.) et la pondération temporelle."""
        from quant_core import MLTrader, add_features, join_macro_data

        df_raw = _make_ohlcv_df(base_price=100.0, n_days=120)
        df_feat = add_features(df_raw, labeling_mode="triple_barrier")
        dates = df_raw.index
        macro_df = pd.DataFrame({
            "SPY_Return": np.zeros(len(dates)),
            "SPY_Ret_5d": np.zeros(len(dates)),
            "SPY_Ret_20d": np.zeros(len(dates)),
            "VIX": np.linspace(15.0, 22.0, len(dates)),
            "TNX": np.full(len(dates), 4.2),
            "DXY_Return": np.zeros(len(dates)),
            "VIX_Term_Ratio": np.ones(len(dates)),
            "Yield_Spread": np.full(len(dates), 0.5),
        }, index=dates)
        df_full = join_macro_data(df_feat, macro_df)

        for col in ["RSI_Change_5", "MACD_Hist", "MACD_Hist_Slope", "DI_Diff", "ER_10",
                    "Donchian_Pos_20", "Downside_Vol_20", "Pos_52w", "VIX_Change_5d"]:
            self.assertIn(col, df_full.columns, f"Nouvelle feature {col} manquante.")

        # Dernière bougie t strictement conservée avec Target NaN
        self.assertEqual(df_full.index[-1], df_raw.index[-1])
        self.assertTrue(np.isnan(df_full["Target"].iloc[-1]))

        # Pondération temporelle croissante (les observations récentes pèsent plus que les anciennes)
        y_mock = pd.Series([0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
        w = MLTrader._compute_sample_weights(y_mock, min_decay_weight=0.45)
        self.assertEqual(len(w), len(y_mock))
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=4)
        self.assertGreater(w[-1], w[1], "Le poids de la classe 1 récente doit dépasser celui de la classe 1 ancienne.")

    def test_09_wfa_quality_filter_and_adaptive_regime_sizing(self):
        """Vérifie le filtre Qualité WFA (Point 1) et le Sizing Adaptatif SPY/VIX (Point 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "state.json")
            save_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=pf_path, allow_reset=True)

            # GOOD : PF 1.30 / Sharpe +0.35 -> doit passer
            # BAD  : Prob 0.85 mais PF 0.65 / Sharpe -0.30 -> doit être bloquée par Filtre Qualité WFA !
            stubs = {
                "MA": _StubTrader(prob_value=0.60, sharpe=0.35, pf=1.30),
                "MCO": _StubTrader(prob_value=0.85, sharpe=-0.30, pf=0.65),
            }

            def mock_fetch(sym, period, interval):
                return _make_ohlcv_df(base_price=100.0, final_price=100.0)

            # Régime défensif : VIX = 24.0 (>= 20) et SPY < SMA200 (0.0) -> taille divisée par 2 (15% -> 7.5%)
            def mock_macro_defensive(period, interval):
                dates = pd.date_range("2025-01-01", periods=80, freq="B")
                return pd.DataFrame(
                    {"VIX": np.full(80, 24.0), "SPY_Above_SMA200": np.zeros(80)},
                    index=dates,
                )

            engine = AutoTraderEngine(
                data_fetcher=mock_fetch,
                macro_fetcher=mock_macro_defensive,
                price_fetcher=lambda s: 100.0,
                state_file=state_path,
            )
            engine.config = AutoTraderConfig(
                tickers=["MA", "MCO"],
                max_open_positions=5,
                position_size_pct=15.0,
                min_buy_prob=0.52,
                use_rsi_shield=False,
                use_wfa_quality_filter=True,
                min_wfa_profit_factor=1.0,
                min_wfa_sharpe=0.0,
                use_adaptive_regime_sizing=True,
                defensive_size_multiplier=0.50,
                portfolio_path=pf_path,
            )
            engine.register_models(stubs)

            rep = engine.run_single_cycle()
            self.assertEqual(len(rep["buys"]), 1, "Seule MA doit être achetée ; MCO doit être filtrée par Qualité WFA.")
            self.assertEqual(rep["buys"][0]["symbol"], "MA")
            # En régime défensif (x0.50), allocation = 7.5% de 100 000$ = 7 500$ -> ~74 actions à 100$ (et non 149)
            self.assertLessEqual(rep["buys"][0]["qty"], 75)
            self.assertGreaterEqual(rep["buys"][0]["qty"], 70)

            st_snap = engine.get_status()
            self.assertIn("DÉFENSIF", st_snap["market_regime"])
            self.assertAlmostEqual(st_snap["effective_position_pct"], 7.5, places=1)
            mco_row = next(r for r in st_snap["last_rankings"] if r["symbol"] == "MCO")
            self.assertIn("QUALITÉ WFA", mco_row["status"])

    def test_10_state_persistence_and_restore_after_restart(self):
        """Vérifie que les transactions (trade_events, logs, PnL) sont restaurées après régénération de l'appli."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf_path = os.path.join(tmpdir, "pf.json")
            state_path = os.path.join(tmpdir, "auto_state.json")
            save_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=pf_path, allow_reset=True)

            # Vérifier aussi les nouvelles valeurs par défaut (capture utilisateur)
            d = AutoTraderConfig()
            self.assertEqual(d.scan_interval_sec, 3600)
            self.assertEqual(d.min_wfa_profit_factor, 0.90)
            self.assertEqual(d.min_wfa_sharpe, 0.0)
            self.assertEqual(d.min_buy_prob, 0.52)

            engine1 = AutoTraderEngine(
                data_fetcher=lambda s, p, i: _make_ohlcv_df(100.0, final_price=100.0),
                macro_fetcher=lambda p, i: None,
                price_fetcher=lambda s: 100.0,
                state_file=state_path,
            )
            engine1.config = AutoTraderConfig(
                tickers=["AAPL"],
                use_rsi_shield=False,
                portfolio_path=pf_path,
            )
            engine1.register_models({"AAPL": _StubTrader(prob_value=0.70, sharpe=0.30, pf=1.25)})
            engine1.run_single_cycle()
            self.assertEqual(engine1.get_status()["total_buys"], 1)

            # Simule un redémarrage complet / régénération de l'application (nouvelle instance)
            engine2 = AutoTraderEngine(state_file=state_path)
            st2 = engine2.get_status()
            self.assertEqual(st2["total_buys"], 1, "Le compteur d'achats doit être restauré après redémarrage.")
            self.assertEqual(len(st2["trade_events"]), 1, "Les transactions Autopilot doivent être restaurées.")
            self.assertEqual(st2["trade_events"][0]["ticker"], "AAPL")
            self.assertGreaterEqual(st2["cycle_count"], 1)


if __name__ == "__main__":
    unittest.main()
