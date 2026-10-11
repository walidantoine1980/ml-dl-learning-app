# XGBoost Stock Predictor Pro 📈
*Terminal Quantitatif Institutionnel, Ensemble ML & Trading Automatisé*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20RandomForest-green.svg)](https://xgboost.readthedocs.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20%26%20Modular%20(quant__core)-orange.svg)](#architecture-du-projet)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(7%2F7)-brightgreen.svg)](#-suite-de-tests-automatis%C3%A9s)

Un terminal quantitatif complet conçu selon les standards de l'industrie financière. Le système fusionne le **Machine Learning d'Ensemble (XGBoost, LightGBM, Random Forest)**, l'analyse technique stationnaire, le **Traitement du Langage Naturel (NLP)** et le **Pricing d'Options (Black-Scholes & Greeks)** pour générer des signaux de trading robustes et des stratégies de couverture.

Le système intègre un simulateur de **Paper Trading local** 100% autonome avec gestion des liquidités, slippage, commissions et **vrais Stops dynamiques ATR**.

---

## 🏛️ Architecture du Projet (Clean Two-Tier Architecture)

Le dépôt est strictement découplé en deux couches pour garantir fiabilité et indépendance :

1.  **`quant_core.py` (Moteur Quantitatif Pur)** :
    *   Module Python indépendant de toute interface graphique (sans aucune dépendance à Streamlit).
    *   Pipeline de données propre : conservation rigoureuse de la dernière bougie $t$ pour les prédictions en direct (zéro décalage temporel).
    *   Ingénierie de caractéristiques stationnaires : ratios invariants d'échelle (`SMA_Ratio`, `NATR = ATR / Close`, `VWAP_Ratio`, `DXY_Return`).
    *   Moteur de backtest événementiel réaliste avec slippage ($0.10\%$), commission ($0.05\%$), coupe-circuit VIX ($> 30$) et **Stops ATR dynamiques ($1.5 \times \text{ATR}$ Stop-Loss et $3.0 \times \text{ATR}$ Take-Profit)**.
    *   Allocation de portefeuille bayésienne de **Black-Litterman**.
    *   Pricing analytique d'options européennes de **Black-Scholes**, Greeks complets et solveur de Brent pour Delta cible.
    *   Persistance atomique du portefeuille (`portfolio.json` et `options_portfolio.json`).

2.  **`xgboost_trader.py` (Terminal Web Interactif)** :
    *   Interface web riche développée avec Streamlit.
    *   Visualisation Plotly interactive (Bandes de Bollinger, RSI, MACD, Volume).
    *   Explicabilité des décisions en temps réel par **SHAP Waterfall**.
    *   Tearsheets quantitatifs, simulation de Monte Carlo (Bootstrap) et académie pédagogique.

3.  **`headless_bot.py` (Démon Autonome d'Arrière-Plan)** :
    *   Démon 24h/24 piloté par `schedule` ou exécutable en une passe (`--once`).
    *   Surveillance proactive des positions ouvertes : exécution automatique des Stop-Loss et Take-Profit en séance.
    *   Enrichissement nocturne post-clôture et alertes formatées via **Discord** et **Telegram**.

---

## 🚀 Fonctionnalités Principales

*   **Terminal de Trading (Machine Learning)** :
    *   Modèle d'ensemble combinant **XGBoost Classifier**, **LightGBM** et **Random Forest** en vote probabiliste pondéré (*soft voting*).
    *   **Features Stationnaires** : Ratios invariants d'échelle (SMA Ratios, EMA Ratios, Bollinger %B, VWAP Ratio, NATR, Lags de Rendements).
    *   **Walk-Forward Analysis (WFA)** : Validation sur fenêtres glissantes d'entraînement et de test pour quantifier l'érosion d'alpha.
    *   **Auto-Optimisation Bayésienne** : Réglage fin des hyperparamètres via **Optuna** avec `TimeSeriesSplit` (sans fuite temporelle).
*   **Double Filtre de Sécurité Macro & Tendance** :
    *   **Filtre Tendance (SMA 200)** : Détection du régime haussier (*Bull*) vs contre-tendance (*Bear*).
    *   **Coupe-Circuit VIX** : Passage automatique à 100% de liquidités (*Cash*) dès que l'indice de volatilité VIX dépasse 30.
*   **Gestion du Risque Institutionnelle** :
    *   **Critère de Kelly (Half-Kelly)** calculé sur les transactions fermées et plafonné à 5% maximum par position.
    *   **Stops Dynamiques ATR** : Stop-Loss à $1.5 \times \text{ATR}$ et Take-Profit à $3.0 \times \text{ATR}$ (Ratio Risque/Rendement $1:2$) exécutés en simulation et en paper trading.
    *   **Value at Risk (VaR 95%) & CVaR** (Expected Shortfall).
    *   **Simulation de Monte Carlo (Bootstrap)** : 1 000 trajectoires aléatoires pour quantifier la probabilité de ruine.
*   **Allocation de Portefeuille (Black-Litterman)** :
    *   Prior d'équilibre en Parité des Risques (*Inverse Volatility Weighting*).
    *   Ajustement bayésien par les convictions probabilistes de l'IA XGBoost.
*   **Module de Pricing & Backtest d'Options (Black-Scholes)** :
    *   Pricing analytique avec Greeks ($\Delta$, $\Gamma$, $\Theta$, $\mathcal{V}$) vérifié à la machine epsilon (parité Call-Put $0.0$).
    *   Recherche automatique de Strike par Delta cible (solveur de Brent).
    *   Backtester systématique de stratégies vendeuses de prime (*Credit Spreads*, *Iron Condors* à Delta 0.15) avec prise en compte du spread bid-ask et des commissions.
*   **Paper Trading Local Intégré** :
    *   Portefeuille virtuel de 100 000 $ (`portfolio.json`) avec suivi individuel des Stop-Loss et Take-Profit par position.

---

## 📦 Installation & Démarrage Rapide

### 1. Cloner le projet et installer les dépendances
```bash
git clone https://github.com/votre-compte/xgboost-trading.git
cd xgboost-trading
pip install -r requirements.txt
```

### 2. Télécharger les corpus NLP
```bash
python -m textblob.download_corpora
```

### 3. Valider l'intégrité avec la suite de tests
```bash
python tests/test_system.py
```
*Toutes les assertions mathématiques et d'intégrité temporelle doivent afficher `OK`.*

### 4. Lancer l'application Web Streamlit
```bash
streamlit run xgboost_trader.py
```
L'interface s'ouvre sur `http://localhost:8501`.

---

## 🤖 Lancer le Bot Autonome (Headless Daemon)

Le bot peut s'exécuter en tâche de fond pour analyser vos actions à la clôture des marchés et transmettre les alertes :

### Mode planifié quotidien (24h/24) :
```bash
python headless_bot.py
```

### Mode exécution immédiate (pour test ou tâche Cron) :
```bash
python headless_bot.py --once
```

Les webhooks Discord et tokens Telegram se configurent directement dans l'onglet **🤖 Paramétrage du Bot (Headless)** de l'application web ou dans `bot_config.json`.

---

## 🧪 Suite de Tests Automatisés

Le package inclut un banc d'essai complet [`tests/test_system.py`](file:///H:/Mon%20Drive/Antoine-Drive/Antigravity/XGBoost_Trading_Github_Package/XGBoost-Trader-10_Oct_26/XGBoost-Trader-10_Oct_26_Clean/tests/test_system.py) couvrant :

*   `test_01_feature_engineering_integrity` : Non-suppression de la dernière bougie $t$ et calcul du NATR.
*   `test_02_ml_trader_training_and_atr_stops` : Entraînement de l'ensemble et exécution événementielle.
*   `test_03_black_scholes_analytical_pricing` : Parité Call-Put et cohérence des Greeks.
*   `test_04_brent_strike_finder` : Convergence du solveur de Brent vers le Delta cible.
*   `test_05_black_litterman_allocation` : Normalisation à 100% et contrainte Long-Only.
*   `test_06_paper_trading_execution` : Traçabilité des ordres et des stops ATR.
*   `test_07_ticker_conversion_and_multipliers` : Conversion multi-bourses et législation européenne vs américaine.

Pour lancer les tests :
```bash
python tests/test_system.py
```

---

## 🐳 Déploiement via Docker

L'application est conteneurisée et prête pour Dokploy, Coolify, Render, Railway ou VPS :

```bash
# Construction de l'image
docker build -t xgboost-trader .

# Lancement avec persistance des volumes
docker run -d -p 8501:8501 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  --name quant-trader \
  xgboost-trader
```

---

## ⚠️ Avertissement Légal
Ce projet est conçu à des fins d'analyse quantitative, d'apprentissage et de recherche financière. Le trading d'actions et d'options comporte des risques significatifs de perte en capital. Testez toujours vos stratégies en simulation (Paper Trading) avant toute prise de décision sur un compte réel.
