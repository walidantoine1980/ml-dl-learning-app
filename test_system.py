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
    get_black_litterman_weights, execute_equity_trade,
    convert_google_to_yahoo_ticker, get_option_multiplier_and_legislation
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


if __name__ == "__main__":
    unittest.main()
