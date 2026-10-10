# XGBoost Stock Predictor Pro 📈

Un terminal quantitatif complet construit avec Python et Streamlit. L'application fusionne le **Machine Learning d'Ensemble (XGBoost, LightGBM, Random Forest)**, l'analyse technique mathématique avancée, le **Traitement du Langage Naturel (NLP)** et le **Pricing d'Options (Black-Scholes & Greeks)** pour générer des signaux de trading et des stratégies de couverture.

Le système intègre un simulateur de **Paper Trading local** 100% autonome avec gestion des liquidités, slippage et commissions réalistes.

---

## 🚀 Fonctionnalités Principales

*   **Terminal de Trading (Machine Learning)** :
    *   Modèle d'ensemble combinant **XGBoost Classifier**, **LightGBM** et **Random Forest** en vote probabiliste pondéré (*soft voting*).
    *   **Features Stationnaires** : Ratios invariants d'échelle (SMA Ratios, EMA Ratios, Bollinger %B, VWAP Ratio, Lags de Rendements) adaptés aux arbres de décision.
    *   **Walk-Forward Analysis (WFA)** : Validation sur fenêtres glissantes d'entraînement et de test pour éliminer tout surapprentissage.
    *   **Auto-Optimisation Bayésienne** : Réglage fin des hyperparamètres via **Optuna** avec `TimeSeriesSplit` (sans fuite temporelle).
*   **Double Filtre de Sécurité Macro & Tendance** :
    *   **Filtre Tendance (SMA 200)** : Détection du régime haussier (*Bull*) vs contre-tendance (*Bear*).
    *   **Coupe-Circuit VIX** : Passage automatique à 100% de liquidités (*Cash*) dès que l'indice de volatilité VIX dépasse 30.
*   **Filtre de Sentiment NLP (TextBlob & FinBERT)** :
    *   Analyse en direct des actualités financières pour détecter les divergences entre les mathématiques du cours et le sentiment médiatique.
*   **Gestion du Risque Institutionnelle** :
    *   **Critère de Kelly (Half-Kelly)** plafonné à 5% maximum du capital par position.
    *   **Stops Dynamiques ATR** : Stop-Loss à $1.5 \times \text{ATR}$ et Take-Profit à $3.0 \times \text{ATR}$ (Ratio Risque/Rendement $1:2$).
    *   **Value at Risk (VaR 95%) & CVaR** (Expected Shortfall).
    *   **Simulation de Monte Carlo (Bootstrap)** : 1 000 trajectoires aléatoires pour quantifier la probabilité de ruine.
*   **Allocation de Portefeuille (Black-Litterman)** :
    *   Prior d'équilibre en Parité des Risques (*Inverse Volatility Weighting*).
    *   Ajustement bayésien par les convictions probabilistes de l'IA XGBoost.
*   **Module de Pricing & Backtest d'Options (Black-Scholes)** :
    *   Pricing analytique sans risque de division par zéro, Greeks ($\Delta$, $\Gamma$, $\Theta$, $\mathcal{V}$) et graphiques de Payoff.
    *   Recherche automatique de Strike par Delta cible (solveur de Brent).
    *   Backtester systématique de stratégies vendeuses de prime (*Credit Spreads*, *Iron Condors* à Delta 0.15).
*   **Paper Trading Local Intégré** :
    *   Portefeuille virtuel de 100 000 $ (`portfolio.json`) avec calcul du slippage (0.10%) et des commissions (0.05%) et carnet de transactions.
    *   Portefeuille d'options virtuel indépendant (`options_portfolio.json`) avec autopsie quotidienne du cycle de vie des contrats.
*   **Démon Autonome (`headless_bot.py`)** :
    *   Script d'arrière-plan quotidien qui télécharge l'historique post-clôture, réentraîne les modèles, exécute les ordres virtuels et transmet les alertes via **Telegram** et **Discord**.
*   **Académie Interactive Intégrée** :
    *   Modules pédagogiques détaillant le fonctionnement de XGBoost, les indicateurs techniques, Black-Scholes et l'histoire de la physique appliquée à la finance quantitative.

---

## 📦 Installation & Déploiement Local

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

### 3. Lancer l'application Streamlit
```bash
streamlit run xgboost_trader.py
```
L'interface s'ouvre sur `http://localhost:8501`.

---

## 🤖 Lancer le Bot Autonome (Headless Daemon)

Le bot peut s'exécuter 24h/24 en tâche de fond pour analyser vos actions à la clôture des marchés et envoyer des signaux :
```bash
python headless_bot.py
```
Les webhooks Discord et tokens Telegram se configurent directement dans l'onglet **🤖 Paramétrage du Bot (Headless)** de l'application web.

---

## 🐳 Déploiement via Docker

L'application est entièrement conteneurisée et prête pour Dokploy, Coolify, Render, Railway ou VPS :

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
Ce projet est conçu à des fins d'analyse quantitative, d'apprentissage et de recherche. Le trading d'actions et d'options comporte des risques de perte en capital. Testez toujours vos stratégies en simulation (Paper Trading) avant toute décision financière.
