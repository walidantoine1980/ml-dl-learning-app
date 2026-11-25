"""
===============================================================================
SUITE DE TESTS UNITAIRES ET D'INTÉGRATION QUANTITATIVE
XGBoost Stock Predictor Pro - Test Suite
===============================================================================
Exécution :
    python tests/test_system.py
    ou
    pytest tests/
===============================================================================
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ajout du chemin du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_core import (
    add_features, MLTrader, black_scholes, find_strike_for_delta,
    get_black_litterman_weights, execute_equity_trade, execute_option_trade,
    convert_google_to_yahoo_ticker, get_option_multiplier_and_legislation,
    load_portfolio, save_portfolio, reset_portfolio,
    load_options_portfolio, save_options_portfolio, reset_options_portfolio,
    evaluate_multi_regime_triggers
)


class TestQuantCore(unittest.TestCase):
    """Banc d'essai de validation pour le moteur quantitatif."""

    def setUp(self):
        """Génère un jeu de données synthétique réaliste (Geometric Brownian Motion)."""
        np.random.seed(42)
        n_days = 250
        dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
        
        # Simulation d'un cours avec dérive positive et volatilité
        dt = 1 / 252
        mu = 0.10
        sigma = 0.20
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n_days)
        prices = 100.0 * np.exp(np.cumsum(returns))
        
        high = prices * (1.0 + np.abs(np.random.normal(0, 0.01, size=n_days)))
        low = prices * (1.0 - np.abs(np.random.normal(0, 0.01, size=n_days)))
        open_p = (high + low) / 2.0
        volume = np.random.randint(500000, 2000000, size=n_days)
        
        self.sample_df = pd.DataFrame({
            'Open': open_p,
            'High': high,
            'Low': low,
            'Close': prices,
            'Volume': volume
        }, index=dates)

    def test_01_feature_engineering_integrity(self):
        """Vérifie que la dernière bougie t n'est JAMAIS perdue et que NATR est calculé."""
        df_feat = add_features(self.sample_df)
        
        # Vérification 1 : La dernière date doit être strictement identique
        self.assertEqual(
            self.sample_df.index[-1], df_feat.index[-1],
            "La dernière bougie t (Close d'aujourd'hui) ne doit pas être supprimée par add_features !"
        )
        
        # Vérification 2 : Présence des indicateurs clés
        self.assertIn('NATR', df_feat.columns, "NATR (Normalized ATR) doit être présent.")
        self.assertIn('SMA_20_Ratio', df_feat.columns, "SMA_20_Ratio doit être présent.")
        self.assertIn('VWAP_Ratio', df_feat.columns, "VWAP_Ratio doit être présent.")
        self.assertIn('Target', df_feat.columns, "Target doit être présent.")
        
        # Vérification 3 : Seule la dernière ligne doit avoir Target = NaN
        self.assertTrue(np.isnan(df_feat['Target'].iloc[-1]), "La cible de la dernière bougie t doit être NaN.")
        self.assertFalse(df_feat['Target'].iloc[:-1].isna().any(), "Toutes les cibles passées doivent être connues.")

    def test_02_ml_trader_training_and_atr_stops(self):
        """Vérifie l'entraînement de l'ensemble ML et l'exécution des stops ATR."""
        df_feat = add_features(self.sample_df)
        trader = MLTrader()
        features = trader.train(df_feat, optimize=False, use_wfa=False)
        
        self.assertTrue(trader.is_trained, "Le modèle doit être marqué entraîné.")
        self.assertGreater(trader.accuracy, 0.0, "L'accuracy doit être positive.")
        self.assertIsNotNone(trader.backtest_results, "Le backtest doit contenir des résultats.")
        
        # Prédiction sur la dernière bougie t
        last_row = df_feat.iloc[-1:]
        prob = trader.predict(last_row, features)
        self.assertIsNotNone(prob, "La prédiction ne doit pas être None.")
        self.assertGreaterEqual(prob, 0.0, "La probabilité doit être >= 0.")
        self.assertLessEqual(prob, 1.0, "La probabilité doit être <= 1.")

    def test_03_black_scholes_analytical_pricing(self):
        """Vérifie l'exactitude mathématique de Black-Scholes et la parité Call-Put."""
        S, K, T, r, sigma = 100.0, 100.0, 30/365.0, 0.05, 0.20
        
        c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(S, K, T, r, sigma, "call")
        p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(S, K, T, r, sigma, "put")
        
        # Parité Call-Put : C - P = S - K * exp(-r*T)
        parity_lhs = c_price - p_price
        parity_rhs = S - K * np.exp(-r * T)
        self.assertAlmostEqual(parity_lhs, parity_rhs, places=5, msg="La parité Call-Put doit être respectée.")
        
        # Greeks
        self.assertGreater(c_delta, 0.0, "Le Delta du Call doit être positif.")
        self.assertLess(p_delta, 0.0, "Le Delta du Put doit être négatif.")
        self.assertAlmostEqual(c_gamma, p_gamma, places=6, msg="Le Gamma Call et Put doit être identique.")
        self.assertAlmostEqual(c_vega, p_vega, places=6, msg="Le Vega Call et Put doit être identique.")

    def test_04_brent_strike_finder(self):
        """Vérifie que le solveur de Brent retrouve le Strike exact pour un Delta cible."""
        S, T, r, sigma = 150.0, 45/365.0, 0.05, 0.25
        target_delta = 0.15
        
        k_found = find_strike_for_delta(S, T, r, sigma, "call", target_delta)
        _, computed_delta, _, _, _ = black_scholes(S, k_found, T, r, sigma, "call")
        
        self.assertAlmostEqual(computed_delta, target_delta, places=3, msg="Le solveur de Brent doit converger vers le delta cible.")

    def test_05_black_litterman_allocation(self):
        """Vérifie l'allocation bayésienne de Black-Litterman."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        np.random.seed(42)
        rets_df = pd.DataFrame(np.random.normal(0.0005, 0.015, size=(100, 3)), columns=tickers)
        market_weights = {'AAPL': 0.333, 'MSFT': 0.333, 'GOOGL': 0.334}
        views = {'AAPL': 0.05, 'MSFT': -0.02, 'GOOGL': 0.00}
        
        bl_weights = get_black_litterman_weights(rets_df, market_weights, views)
        
        total_w = sum(bl_weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=4, msg="La somme des poids Black-Litterman doit être égale à 100%.")
        for t in tickers:
            self.assertGreaterEqual(bl_weights[t], 0.0, f"Le poids de {t} doit être positif (Long-Only).")

    def test_06_paper_trading_execution(self):
        """Vérifie l'exécution d'un trade virtuel et le dimensionnement des stops ATR."""
        pf = {"cash": 50000.0, "positions": {}, "history": []}
        
        # Achat
        success, msg = execute_equity_trade(pf, "AAPL", 10, "BUY", 200.0, current_atr=4.0)
        self.assertTrue(success, "L'achat doit réussir.")
        self.assertIn("AAPL", pf["positions"], "AAPL doit être dans les positions.")
        self.assertLess(pf["cash"], 50000.0, "Le cash doit diminuer.")
        
        pos = pf["positions"]["AAPL"]
        self.assertIn("stop_loss", pos, "Le stop-loss doit être enregistré dans la position.")
        self.assertIn("take_profit", pos, "Le take-profit doit être enregistré dans la position.")
        
        # Vente
        success_sell, msg_sell = execute_equity_trade(pf, "AAPL", 10, "SELL", 210.0)
        self.assertTrue(success_sell, "La vente doit réussir.")
        self.assertNotIn("AAPL", pf["positions"], "La position soldée doit être retirée.")
        self.assertGreater(pf["cash"], 50000.0, "Le cash doit avoir augmenté après plus-value.")

    def test_07_ticker_conversion_and_multipliers(self):
        """Vérifie la conversion Google -> Yahoo et la législation des options."""
        self.assertEqual(convert_google_to_yahoo_ticker("EPA:MC"), "MC.PA")
        self.assertEqual(convert_google_to_yahoo_ticker("NASDAQ:AAPL"), "AAPL")
        
        mult_us, leg_us = get_option_multiplier_and_legislation("AAPL")
        self.assertEqual(mult_us, 100)
        
        mult_eu, leg_eu = get_option_multiplier_and_legislation("MC.PA")
        self.assertEqual(mult_eu, 10)

    def test_08_walk_forward_analysis_length_mismatch(self):
        """Vérifie que la WFA s'exécute sans erreur même si data et valid_train ont des longueurs divergentes."""
        # Création d'un dataset multi-années
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        df = pd.DataFrame({
            'Open': 100.0 + np.random.randn(1000).cumsum(),
            'High': 102.0 + np.random.randn(1000).cumsum(),
            'Low': 98.0 + np.random.randn(1000).cumsum(),
            'Close': 100.0 + np.random.randn(1000).cumsum(),
            'Volume': 1000000 + np.random.randint(0, 50000, size=1000)
        }, index=dates)
        
        df_feat = add_features(df)
        # Injection volontaire de plusieurs NaN au milieu et à la fin pour tester la robustesse des tailles
        df_feat.iloc[100:110, df_feat.columns.get_loc('Target')] = np.nan
        
        trader = MLTrader()
        features = trader.train(df_feat, optimize=False, use_wfa=True, wfa_train_window="2Y", wfa_step="6M")
        self.assertTrue(trader.is_trained, "La WFA doit s'entraîner sans ValueError de longueur.")
        self.assertIsNotNone(trader.backtest_results, "Le backtest WFA doit contenir des résultats.")

    def test_09_options_portfolio_and_execution(self):
        """Vérifie l'intégrité du portefeuille d'options (dict vs list) et l'achat/revente de contrats."""
        pf_opt = load_options_portfolio("non_existent_file_test.json")
        self.assertIsInstance(pf_opt, dict, "Le portefeuille d'options doit être un dictionnaire.")
        self.assertIsInstance(pf_opt["positions"], dict, "positions doit impérativement être un dict (et JAMAIS une liste).")
        self.assertIsInstance(pf_opt["history"], list, "history doit être une liste.")
        self.assertEqual(pf_opt["cash"], 100000.0, "Le solde initial d'options doit être 100 000 $.")
        
        # Test Achat Option CALL (Norme US : x100)
        success, contract_id, msg = execute_option_trade(
            pf_opt, ticker="AAPL", opt_type="call", strike=150.0,
            days_to_expiry=30, qty=2, premium=5.0, underlying_price=150.0, action="BUY"
        )
        self.assertTrue(success, "L'achat d'option doit réussir.")
        self.assertIsNotNone(contract_id, "Un contract_id doit être généré.")
        self.assertIn(contract_id, pf_opt["positions"], "Le contrat doit être stocké par sa clé contract_id.")
        # Coût = 2 contrats * 5.0 * 100 = 1000.0 $
        self.assertEqual(pf_opt["cash"], 99000.0, "Le cash doit être débité de 1000 $.")
        
        # Test Revente Option CALL
        success_sell, sold_id, msg_sell = execute_option_trade(
            pf_opt, ticker="AAPL", opt_type="call", strike=150.0,
            days_to_expiry=20, qty=2, premium=7.0, underlying_price=155.0, action="SELL"
        )
        self.assertTrue(success_sell, "La revente d'option doit réussir.")
        self.assertNotIn(contract_id, pf_opt["positions"], "La position d'option soldée doit être purgée.")
        # Recette = 2 contrats * 7.0 * 100 = 1400.0 $ -> Cash final = 99000 + 1400 = 100400 $
        self.assertEqual(pf_opt["cash"], 100400.0, "Le cash doit inclure le produit de revente et la plus-value.")

    def test_10_portfolio_resilience_to_corrupted_json(self):
        """Vérifie la résilience du chargeur si un fichier JSON contenait 'positions': [] (ancienne anomalie)."""
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            # Simule l'ancienne structure erronée où positions était une liste vide
            json.dump({"cash": 50000.0, "positions": [], "history": []}, tf)
            temp_path = tf.name
            
        try:
            # Doit réparer et normaliser automatiquement en dictionnaire
            loaded = load_options_portfolio(temp_path)
            self.assertIsInstance(loaded["positions"], dict, "Un 'positions': [] doit être converti en {} dict.")
            self.assertEqual(loaded["cash"], 50000.0)
            
            # Vérifier qu'on peut immédiatement indexer une string sans TypeError
            loaded["positions"]["TEST_CONTRACT"] = {"qty": 1}
            self.assertEqual(loaded["positions"]["TEST_CONTRACT"]["qty"], 1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_11_multi_regime_triggers(self):
        """Vérifie l'évaluation rigoureuse des déclencheurs multi-régimes (RSI Shield, VIX Breaker, Squeeze)."""
        row_bull = pd.Series({
            'Close': 150.0,
            'SMA_200': 130.0,
            'SMA_50': 140.0,
            'RSI': 80.0, # Surchat extrême
            'Volume_Ratio': 2.0, # Volume Surge
            'BB_Width': 0.04, # Bollinger Squeeze
            'BB_Upper': 155.0,
            'BB_Lower': 145.0,
            'VIX': 15.0
        })
        triggers = evaluate_multi_regime_triggers(row_bull, current_price=150.0, current_atr=3.0, trader_prob=0.70)
        
        self.assertEqual(triggers['trend_regime']['status'], 'BULLISH')
        self.assertEqual(triggers['rsi_shield']['status'], 'OVERBOUGHT_SHIELD')
        self.assertEqual(triggers['volume_surge']['status'], 'INSTITUTIONAL_INFLOW')
        self.assertEqual(triggers['bollinger_squeeze']['status'], 'SQUEEZE_ACTIVE')
        self.assertEqual(triggers['macro_vix']['status'], 'CALM_MARKET')
        self.assertAlmostEqual(triggers['risk_levels']['breakeven_trigger'], 154.5)
        
        # Test Coupe-Circuit VIX > 30
        row_crisis = pd.Series({
            'Close': 100.0,
            'SMA_200': 110.0,
            'SMA_50': 105.0,
            'RSI': 25.0,
            'VIX': 35.0
        })
        trig_crisis = evaluate_multi_regime_triggers(row_crisis, current_price=100.0, current_atr=2.0)
        self.assertEqual(trig_crisis['macro_vix']['status'], 'CIRCUIT_BREAKER_ON')
        self.assertEqual(trig_crisis['trend_regime']['status'], 'BEARISH')
        self.assertEqual(trig_crisis['rsi_shield']['status'], 'OVERSOLD_OPPORTUNITY')

    def test_12_global_stock_universe(self):
        """Vérifie la présence et la conversion des actions mondiales majeures (Asie, Europe, US, LatAm)."""
        from tickers_db import MAJOR_STOCKS, PREDEFINED_PORTFOLIOS
        
        # Vérification des nouveaux champions internationaux
        self.assertIn("Taiwan Semiconductor (TSMC) (TW)", MAJOR_STOCKS)
        self.assertIn("ASML Holding (NL)", MAJOR_STOCKS)
        self.assertIn("Novo Nordisk (DK)", MAJOR_STOCKS)
        self.assertIn("Alibaba Group (CN)", MAJOR_STOCKS)
        self.assertIn("Tencent Holdings (CN)", MAJOR_STOCKS)
        self.assertIn("MercadoLibre (LATAM)", MAJOR_STOCKS)
        
        # Vérification des conversions Yahoo Finance
        self.assertEqual(convert_google_to_yahoo_ticker(MAJOR_STOCKS["Taiwan Semiconductor (TSMC) (TW)"]), "TSM")
        self.assertEqual(convert_google_to_yahoo_ticker(MAJOR_STOCKS["ASML Holding (NL)"]), "ASML.AS")
        self.assertEqual(convert_google_to_yahoo_ticker(MAJOR_STOCKS["Tencent Holdings (CN)"]), "0700.HK")
        self.assertEqual(convert_google_to_yahoo_ticker(MAJOR_STOCKS["Novo Nordisk (DK)"]), "NVO")
        
        # Vérification des portefeuilles thématiques mondiaux
        self.assertIn("Leaders Mondiaux Semi-conducteurs & IA", PREDEFINED_PORTFOLIOS)
        self.assertIn("Titans Émergents & FinTech (Asie & LatAm)", PREDEFINED_PORTFOLIOS)
        self.assertIn("Santé Mondiale & Pharma Giants", PREDEFINED_PORTFOLIOS)

    def test_13_academies_integrity(self):
        """Vérifie l'intégrité et la disponibilité des 6 Grandes Académies Quantitatives."""
        import academy_core
        self.assertTrue(callable(getattr(academy_core, "render_indicators_academy", None)))
        self.assertTrue(callable(getattr(academy_core, "render_xgboost_academy", None)))
        self.assertTrue(callable(getattr(academy_core, "render_strategy_academy", None)))
        self.assertTrue(callable(getattr(academy_core, "render_advanced_academy", None)))
        self.assertTrue(callable(getattr(academy_core, "render_options_academy", None)))
        self.assertTrue(callable(getattr(academy_core, "render_news_academy", None)))
    def test_14_mltrader_save_load_features_persistence(self):
        """Vérifie que les features sont fidèlement sauvegardées et rechargées avec MLTrader."""
        import tempfile
        df_feat = add_features(self.sample_df)
        trader = MLTrader()
        features = trader.train(df_feat, optimize=False, use_wfa=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tf:
            temp_path = tf.name
            
        try:
            trader.save(temp_path)
            loaded = MLTrader.load(temp_path)
            self.assertTrue(loaded.is_trained)
            self.assertEqual(loaded.features, features, "Les features rechargées doivent être strictement identiques.")
            self.assertGreater(len(loaded.features), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_15_wfa_short_history_exception(self):
        """Vérifie que la WFA lève une ValueError explicite si l'historique est trop court."""
        df_feat = add_features(self.sample_df) # ~250 jours (1 an)
        trader = MLTrader()
        # Demander une fenêtre de 5 ans sur un historique de 1 an doit lever une ValueError
        with self.assertRaises(ValueError):
            trader.train(df_feat, optimize=False, use_wfa=True, wfa_train_window="5Y", wfa_step="6M")

    def test_16_equity_portfolio_persistence_and_reset_protection(self):
        """Vérifie la mémorisation des actions et la protection contre toute réinitialisation non autorisée."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_path = tf.name
        try:
            pf = {"cash": 100000.0, "positions": {}, "history": []}
            execute_equity_trade(pf, "NVDA", 15, "BUY", 120.0, current_atr=3.0)
            save_portfolio(pf, filepath=temp_path)

            # 1. Vérifier la persistance après rechargement
            reloaded = load_portfolio(filepath=temp_path)
            self.assertIn("NVDA", reloaded["positions"])
            self.assertEqual(reloaded["positions"]["NVDA"]["qty"], 15)

            # 2. Tentative d'écrasement accidentel (sans allow_reset=True) -> DOIT être bloquée
            save_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=temp_path, allow_reset=False)
            protected = load_portfolio(filepath=temp_path)
            self.assertIn("NVDA", protected["positions"], "Le portefeuille d'actions ne doit JAMAIS être réinitialisé sans le bouton dédié.")
            self.assertEqual(protected["positions"]["NVDA"]["qty"], 15)

            # 3. Réinitialisation explicite via reset_portfolio (bouton Réinitialiser le compte) -> DOIT réussir
            reset_portfolio(filepath=temp_path)
            after_reset = load_portfolio(filepath=temp_path)
            self.assertEqual(after_reset["cash"], 100000.0)
            self.assertEqual(len(after_reset["positions"]), 0)
            self.assertEqual(len(after_reset["history"]), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_17_options_portfolio_persistence_and_reset_protection(self):
        """Vérifie la mémorisation des options et la protection contre toute réinitialisation non autorisée."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            temp_path = tf.name
        try:
            pf_opt = {"cash": 100000.0, "positions": {}, "history": []}
            success, cid, _ = execute_option_trade(
                pf_opt, ticker="AAPL", opt_type="call", strike=200.0,
                days_to_expiry=30, qty=3, premium=4.5, underlying_price=198.0, action="BUY"
            )
            self.assertTrue(success)
            save_options_portfolio(pf_opt, filepath=temp_path)

            # 1. Vérifier la persistance après rechargement
            reloaded = load_options_portfolio(filepath=temp_path)
            self.assertIn(cid, reloaded["positions"])
            self.assertEqual(reloaded["positions"][cid]["qty"], 3)

            # 2. Tentative d'écrasement accidentel (sans allow_reset=True) -> DOIT être bloquée
            save_options_portfolio({"cash": 100000.0, "positions": {}, "history": []}, filepath=temp_path, allow_reset=False)
            protected = load_options_portfolio(filepath=temp_path)
            self.assertIn(cid, protected["positions"], "Le portefeuille d'options ne doit JAMAIS être réinitialisé sans le bouton dédié.")
            self.assertEqual(protected["positions"][cid]["qty"], 3)

            # 3. Réinitialisation explicite via reset_options_portfolio (bouton Réinitialiser le compte d'Options) -> DOIT réussir
            reset_options_portfolio(filepath=temp_path)
            after_reset = load_options_portfolio(filepath=temp_path)
            self.assertEqual(after_reset["cash"], 100000.0)
            self.assertEqual(len(after_reset["positions"]), 0)
            self.assertEqual(len(after_reset["history"]), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_18_watchlist_csv_validation_and_translation(self):
        """Vérifie la validation de conformité et la traduction des fichiers Watchlist CSV (Investing.com, Reuters, etc.)."""
        import tempfile
        from tickers_db import (
            normalize_watchlist_symbol,
            parse_and_validate_watchlist_csv,
            save_translated_watchlist_csv,
        )

        # 1. CSV Investing.com / Reuters avec UTF-8 BOM, guillemets, virgules dans les nombres et colonnes dupliquées
        investing_csv = (
            '\ufeff"Name","Symbol","Exchange","Last","Vol.","Daily","Daily"\n'
            '"Akamai Technologies Inc","AKAM.O","NASDAQ","91.70","8,568,715","Buy","8.34%"\n'
            '"HP Inc","HPQ","NYSE","28.66","10,190,530","Sell","-0.17%"\n'
            '"Oracle Corporation","ORCL.K","NYSE","113.89","15,394,703","Sell","-0.97%"\n'
            '"Qnity Electronics Inc","Q","NYSE","115.82","1,873,287","Sell","1.16%"\n'
        ).encode("utf-8")

        res = parse_and_validate_watchlist_csv(investing_csv, "Tech_Investing.csv")
        self.assertTrue(res["is_valid"], f"Le CSV Investing.com doit être valide : {res['errors']}")
        self.assertTrue(res["needs_translation"], "Le CSV virgule avec suffixes .O/.K doit déclencher la traduction.")
        self.assertEqual(len(res["entries"]), 4)
        symbols = [e["symbol"] for e in res["entries"]]
        self.assertEqual(symbols, ["AKAM", "HPQ", "ORCL", "Q"])
        self.assertIn(("AKAM.O", "AKAM"), res["conversions"])
        self.assertIn(("ORCL.K", "ORCL"), res["conversions"])
        self.assertTrue(res["translated_csv"].startswith("Name;Symbol;Exchange\n"))

        # Vérifier la sauvegarde traduite dans un répertoire temporaire
        with tempfile.TemporaryDirectory() as tmpdir:
            ok, saved_path = save_translated_watchlist_csv(res["translated_csv"], res["clean_filename"], watchlists_dir=tmpdir)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "rb") as f:
                reloaded = parse_and_validate_watchlist_csv(f.read(), os.path.basename(saved_path))
            self.assertTrue(reloaded["is_valid"])
            self.assertEqual([e["symbol"] for e in reloaded["entries"]], ["AKAM", "HPQ", "ORCL", "Q"])

        # 2. Traductions internationales / Google Finance / Classes d'actions
        self.assertEqual(normalize_watchlist_symbol("EPA:MC"), "MC.PA")
        self.assertEqual(normalize_watchlist_symbol("NESN.S"), "NESN.SW")
        self.assertEqual(normalize_watchlist_symbol("MC", "Euronext Paris"), "MC.PA")
        self.assertEqual(normalize_watchlist_symbol("BRK.B", "NYSE"), "BRK-B")

        # 3. Liste simple de tickers (1 colonne sans en-tête)
        simple_csv = b"AAPL\nMSFT\nNVDA\n"
        res_simple = parse_and_validate_watchlist_csv(simple_csv, "simple.csv")
        self.assertTrue(res_simple["is_valid"])
        self.assertEqual([e["symbol"] for e in res_simple["entries"]], ["AAPL", "MSFT", "NVDA"])

        # 4. CSV non conforme (sans colonne ticker)
        bad_csv = b"ColA,ColB\nfoo,bar\n"
        res_bad = parse_and_validate_watchlist_csv(bad_csv, "invalid.csv")
        self.assertFalse(res_bad["is_valid"])
        self.assertTrue(len(res_bad["errors"]) > 0)

    def test_19_new_features_and_triple_barrier_labeling(self):
        """Vérifie les nouvelles features stationnaires (Garman-Klass, CMF, MFI, OBV, etc.) et le Triple Barrier Labeling."""
        df_tb = add_features(self.sample_df, labeling_mode="triple_barrier", tb_horizon=5)
        for col in ['GK_Vol_20', 'Vol_Ratio_5_20', 'CLV', 'CMF_20', 'MFI_14', 'OBV_ZScore',
                    'Gap_Open', 'Ret_5d', 'Ret_21d', 'Dist_High_52w', 'Price_ZScore_20',
                    'SMA_50_Slope', 'DayOfWeek_Sin', 'TurnOfMonth']:
            self.assertIn(col, df_tb.columns, f"Feature {col} manquante.")
        self.assertEqual(self.sample_df.index[-1], df_tb.index[-1], "La dernière bougie t doit être conservée.")
        self.assertTrue(np.isnan(df_tb['Target'].iloc[-1]), "Target de la dernière bougie t doit être NaN.")
        past_targets = df_tb['Target'].iloc[:-1]
        self.assertFalse(past_targets.isna().any(), "Toutes les cibles Triple Barrier passées doivent être définies.")
        self.assertTrue(set(past_targets.unique()).issubset({0.0, 1.0}))

    def test_20_backtest_causality_and_advanced_strategy_modes(self):
        """Vérifie l'absence de ré-entrée le même jour, la calibration, le Trailing Stop, Vol Sizing et les métriques Tearsheet."""
        import tempfile
        # 1. Test déterministe : pas de ré-entrée le même jour après Stop-Loss
        dates = pd.date_range("2025-01-01", periods=6, freq="B")
        bt_in = pd.DataFrame({
            'Open':  [100.0, 100.0, 100.0, 90.0, 92.0, 95.0],
            'High':  [101.0, 101.0, 101.0, 91.0, 93.0, 96.0],
            'Low':   [99.0,  99.0,  99.0,  85.0, 91.0, 94.0],
            'Close': [100.0, 100.0, 100.0, 88.0, 92.0, 95.0],
            'ATR':   [2.0,   2.0,   2.0,   2.0,  2.0,  2.0],
            'Prob':  [0.80,  0.80,  0.80,  0.80, 0.80, 0.80],
            'VIX':   [15.0,  15.0,  15.0,  15.0, 15.0, 15.0],
        }, index=dates)
        t_bt = MLTrader()
        res_bt = t_bt.run_event_driven_backtest(bt_in, use_atr_stops=True, atr_sl_mult=1.5, atr_tp_mult=3.0)
        # Entrée jour 1 (index 1, prix ~100.1), Stop Loss touché jour 3 (Open=90 <= SL ~97.1) -> Invested jour 3 DOIT être 0.0 (pas de ré-entrée le jour 3!)
        self.assertEqual(res_bt['Invested'].iloc[3], 0.0, "Aucune ré-entrée ne doit avoir lieu le même jour qu'un Stop-Loss.")

        # 2. Entraînement complet avec Triple Barrier, Calibration, Trailing Stop, Filtres de régime, Vol Sizing & Joblib save/load
        df_tb = add_features(self.sample_df, labeling_mode="triple_barrier")
        trader = MLTrader()
        feats = trader.train(
            df_tb, optimize=False, use_wfa=False,
            calibrate=True, use_trailing_stop=True,
            use_regime_filters=True, use_vol_sizing=True, retrain_final=True
        )
        self.assertTrue(trader.is_trained)
        for k in ['CAGR', 'Profit Factor', 'Calmar Ratio', 'Exposure %', 'Market Sharpe Ratio', 'Sortino Ratio']:
            self.assertIn(k, trader.advanced_metrics, f"Métrique Tearsheet {k} manquante.")

        prob = trader.predict(df_tb.iloc[-1:], feats)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            tmp_model = tf.name
        try:
            trader.save(tmp_model)
            reloaded = MLTrader.load(tmp_model)
            self.assertIsNotNone(reloaded.saved_at)
            self.assertEqual(reloaded.features, trader.features)
            prob2 = reloaded.predict(df_tb.iloc[-1:], feats)
            self.assertAlmostEqual(prob, prob2, places=6)
        finally:
            if os.path.exists(tmp_model):
                os.remove(tmp_model)

    def test_21_black_litterman_neutral_views_and_weekly_annualization(self):
        """Vérifie que Black-Litterman préserve l'équilibre sur vue neutre et gère l'annualisation hebdomadaire."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        np.random.seed(7)
        dates_wk = pd.date_range("2022-01-07", periods=100, freq="W-FRI")
        rets_df = pd.DataFrame(np.random.normal(0.002, 0.03, size=(100, 3)), index=dates_wk, columns=tickers)
        market_weights = {'AAPL': 0.30, 'MSFT': 0.40, 'GOOGL': 0.30}
        # Vue positive uniquement sur AAPL, neutre (0.0) sur les autres -> AAPL doit augmenter vs son prior
        bl_w = get_black_litterman_weights(rets_df, market_weights, {'AAPL': 0.08, 'MSFT': 0.0, 'GOOGL': 0.0})
        self.assertAlmostEqual(sum(bl_w.values()), 1.0, places=4)
        self.assertGreater(bl_w['AAPL'], market_weights['AAPL'], "Une vue positive sur AAPL doit augmenter son poids par rapport au prior.")


if __name__ == "__main__":
    unittest.main()



