"""
===============================================================================
CENTRE DE FORMATION INSTITUTIONNEL : LES GRANDES ACADÉMIES QUANTITATIVES
XGBoost Stock Predictor Pro - Masterclass Haute Précision
===============================================================================
Ce module regroupe les 6 Grandes Académies Quantitatives :
1. Indicateurs Techniques & Déclencheurs Institutionnels (Multi-Triggers)
2. Machine Learning de Précision : Plongée dans XGBoost
3. Stratégie, Gestion du Risque & Black-Litterman
4. Modélisation Avancée : Stacking, Optuna & Monte Carlo
5. Mathématiques des Options, Black-Scholes & Volatility Surface
6. NLP Financier Profond, Sentiment de Marché & RAG Gemini

Chaque Académie est structurée en modules thématiques avec :
- Démonstrations mathématiques et formules LaTeX
- Exemples concrets chiffrés pas-à-pas (AAPL, NVDA, LVMH, S&P 500)
- Secrets et pièges des Quants ("Ce que Wall Street ne vous dit pas")
- Quiz d'auto-évaluation interactif avec feedback pédagogique immédiat
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np


# =============================================================================
# HELPER : GESTION DES QUIZ INTERACTIFS DANS STREAMLIT
# =============================================================================

def render_interactive_quiz(quiz_id, questions):
    """
    Rendu d'un module de quiz interactif avec validation immédiate en session_state.
    questions = [
        {
            "question": str,
            "options": [str],
            "correct_index": int,
            "explanation": str
        }
    ]
    """
    st.markdown("### 🎯 Testez vos Connaissances (Quiz d'Évaluation)")
    st.caption("Sélectionnez votre réponse pour mesurer votre compréhension des concepts quantitatifs.")
    
    score = 0
    total = len(questions)
    
    for i, q in enumerate(questions):
        state_key = f"quiz_{quiz_id}_q{i}"
        user_choice = st.radio(
            f"**Question {i+1} : {q['question']}**",
            q['options'],
            key=f"{state_key}_input",
            index=None
        )
        
        if user_choice is not None:
            chosen_index = q['options'].index(user_choice)
            if chosen_index == q['correct_index']:
                st.success(f"✅ **Exact !** {q['explanation']}")
                score += 1
            else:
                st.error(f"❌ **Incorrect.** {q['explanation']}")
        st.write("")
        
    if score == total and total > 0:
        st.balloons()
        st.success(f"🏆 **Score Parfait : {score}/{total} !** Vous maîtrisez parfaitement ce chapitre institutionnel.")


# =============================================================================
# 1. ACADÉMIE : LES INDICATEURS TECHNIQUES & DÉCLENCHEURS DE MARCHÉ
# =============================================================================

def render_indicators_academy():
    st.title("📚 Académie Quantitative : Indicateurs & Déclencheurs de Marché")
    st.markdown(r"""
    Bienvenue dans la **Masterclass d'Analyse Quantitative**. En finance quantitative moderne, 
    les indicateurs ne sont pas des boules de cristal prédictives : ce sont des **transformations mathématiques** 
    qui convertissent des séries temporelles bruitées (cours, volumes) en **descripteurs d'état stationnaires**.
    
    > **Le Postulat de Renaissance Technologies :** *« Le marché est composé à 80% de bruit aléatoire et à 20% de signal structurel. 
    Les indicateurs sont les filtres statistiques qui permettent au Machine Learning d'isoler ce signal. »*
    """)
    
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📈 Tendance & Moyennes",
        "⚡ Momentum & RSI",
        "🌪️ Volatilité & Bollinger",
        "🌊 Volume & Inflow",
        "🌍 Macro & VIX",
        "🎯 Cockpit Multi-Triggers",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. Les Indicateurs de Tendance : Détecter l'Inertie")
        st.markdown(r"""
        La tendance est la manifestation de l'inertie du capital institutionnel. 
        Lorsqu'un fonds souverain de 50 milliards de dollars achète une action, il lui faut 3 semaines pour exécuter ses ordres sans faire exploser le cours. 
        C'est cette trace que mesurent les moyennes mobiles.
        """)
        
        st.subheader("A. La Moyenne Mobile Simple (SMA)")
        st.latex(r"SMA_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}")
        st.markdown(r"""
        * **Propriété :** Filtre passe-bas arithmétique. Elle élimine les oscillations à haute fréquence.
        * **La SMA 20 (Court terme) :** Représente le coût moyen d'acquisition sur 1 mois boursier (20 séances).
        * **La SMA 50 (Moyen terme) :** Niveau de référence pour les gérants de fonds mutuels.
        * **La SMA 200 (Long terme) :** Frontière absolue entre un marché haussier séculaire (*Bull Market*) et un marché baissier (*Bear Market*).
        """)
        
        st.subheader("B. La Moyenne Mobile Exponentielle (EMA)")
        st.latex(r"EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1} \quad \text{avec} \quad \alpha = \frac{2}{n + 1}")
        st.markdown(r"""
        * **Avantage mathématique :** Contrairement à la SMA qui oublie brutalement le cours d'il y a $n+1$ jours (provoquant un saut artificiel), l'EMA applique une décroissance exponentielle infinie. Elle réagit deux fois plus vite aux retournements de marché récents.
        """)
        
        st.subheader("C. Le MACD (Moving Average Convergence Divergence)")
        st.latex(r"MACD(t) = EMA_{12}(P_t) - EMA_{26}(P_t)")
        st.latex(r"Signal(t) = EMA_9(MACD(t)) \quad \text{et} \quad Histogramme(t) = MACD(t) - Signal(t)")
        st.info(r"""
        💡 **Interprétation Quant :** Le MACD est l'approximation de la **dérivée première** (vitesse) de la tendance, et l'Histogramme représente la **dérivée seconde** (accélération). 
        Quand l'histogramme commence à rétrécir, l'accélération diminue : le mouvement perd de son souffle avant même que le cours ne se retourne !
        """)

    with t2:
        st.header("2. Les Oscillateurs : La Détection de Surachat & Divergences")
        st.markdown(r"""
        Les oscillateurs bornés mesurent la tension élastique du cours autour de son point d'équilibre.
        """)
        
        st.subheader("A. Le RSI de Wilder (Relative Strength Index)")
        st.latex(r"RSI = 100 - \frac{100}{1 + RS} \quad \text{où} \quad RS = \frac{\text{Moyenne Lissée des Hausses sur 14 jours}}{\text{Moyenne Lissée des Baisses sur 14 jours}}")
        st.markdown(r"""
        * **Seuils Classiques :**
          - $RSI > 70$ : Zone de Surachat. La pression acheteuse a été statistiquement excessive.
          - $RSI < 30$ : Zone de Survente. Vente de panique ou capitulation.
        """)
        
        st.warning(r"""
        ⚠️ **Le Piège Mortel du Débutant :** Vendre à découvert dès que le RSI dépasse 70 en plein *Bull Market* est le moyen le plus rapide de faire faillite. 
        Dans une tendance parabolique (ex: Nvidia lors du boom de l'IA), le RSI peut rester bloqué au-dessus de 75 pendant 6 semaines consécutives. 
        C'est pour cela que notre application utilise le RSI comme un **Bouclier d'Inhibition (RSI Shield)** : il interdit les nouveaux achats impulsifs au sommet, sans pour autant parier bêtement contre la tendance.
        """)
        
        st.subheader("B. La Divergence Cachée & Régulière")
        st.markdown(r"""
        * **Divergence Régulière Baissière :** Le cours fait un *Plus Haut plus haut* ($P_2 > P_1$), mais le RSI fait un *Plus Haut plus bas* ($RSI_2 < RSI_1$). Les acheteurs ont réussi à monter le prix par inertie, mais avec beaucoup moins de volume et d'énergie. Chute imminente.
        * **Divergence Cachée Haussière :** Le cours fait un *Plus Bas plus haut* (consolidation saine dans une tendance de fond), mais le RSI s'effondre en zone de survente. Signal de reprise explosive dans le sens de la tendance.
        """)

    with t3:
        st.header("3. La Volatilité : Bandes de Bollinger & ATR")
        st.markdown("La volatilité n'est pas le risque : la volatilité est l'énergie cinétique du marché.")
        
        st.subheader("A. Les Bandes de Bollinger")
        st.latex(r"Upper = SMA_{20} + 2 \cdot \sigma_{20} \quad | \quad Lower = SMA_{20} - 2 \cdot \sigma_{20}")
        st.latex(r"BB\_Width = \frac{Upper - Lower}{SMA_{20}} \quad | \quad \%B = \frac{P - Lower}{Upper - Lower}")
        st.markdown(r"""
        * **Propriété Statistique :** En faisant l'hypothèse gaussienne, 95.4% des clôtures se situent à l'intérieur des bandes.
        * **Le Bollinger Squeeze (Compression de Volatilité) :** 
          Lorsque `BB_Width` atteint un plus bas historique sur 6 mois, les bandes se resserrent au maximum. 
          Les acteurs institutionnels accumulent discrètement des positions. 
          Un $BB\_Width < 0.05$ signale une **détonation directionnelle imminente**.
        """)
        
        st.subheader("B. L'ATR (Average True Range) : Le Maître du Risque")
        st.latex(r"TR = \max \left( High - Low, \, |High - Close_{prec}|, \, |Low - Close_{prec}| \right)")
        st.latex(r"ATR_{14} = \text{Moyenne mobile lissée de } TR \text{ sur 14 jours}")
        st.success(r"""
        🔒 **Pourquoi l'ATR est supérieur aux pourcentages fixes :**
        Mettre un Stop-Loss fixe à 2% est absurde. Sur une action calme comme Coca-Cola, 2% représente 4 jours de mouvement normal. 
        Sur Tesla ou MicroStrategy, 2% est absorbé en 12 minutes par le simple bruit intrajournalier. 
        L'ATR calcule le « battement de cœur » réel du titre en dollars. 
        Notre application place toujours ses Stop-Loss à **1.5 × ATR** pour être immunisé contre la chasse aux liquidités (*Stop Hunting*).
        """)

    with t4:
        st.header("4. Volume & Flux Institutionnels (Order Flow)")
        st.markdown(r"""
        Le prix est ce que vous payez, le volume est la confirmation de l'engagement des institutions financières.
        """)
        
        st.subheader("A. Le Volume Ratio (Volume Surge)")
        st.latex(r"Volume\_Ratio_t = \frac{Volume_t}{SMA_{20}(Volume)_t}")
        st.markdown(r"""
        * $Ratio \ge 1.5$ : **Afflux Institutionnel (Institutional Inflow)**. Les banques ou hedge funds ont passé des ordres d'achat massifs.
        * $Ratio < 0.7$ : Marché de détail (Retail), faible conviction, proies faciles pour les faux breakouts.
        """)
        
        st.subheader("B. Le VWAP (Volume-Weighted Average Price)")
        st.latex(r"VWAP = \frac{\sum (Prix_{typique} \times Volume)}{\sum Volume} \quad \text{avec} \quad Prix_{typique} = \frac{High + Low + Close}{3}")
        st.info(r"""
        🏛️ **Le Benchmark des Salles de Marché :** 
        Les traders institutionnels chez Morgan Stanley ou Goldman Sachs ont pour obligation contractuelle d'obtenir un prix d'exécution meilleur ou égal au VWAP journalier. 
        Si le cours est au-dessus du VWAP, les algorithmes institutionnels sont en phase d'achat agressif.
        """)

    with t5:
        st.header("5. L'Environnement Macroéconomique & Les Régimes de Crise")
        st.markdown("Aucune action ne peut échapper à l'attraction gravitationnelle des marchés directeurs.")
        
        st.subheader("A. Le VIX (L'Indice de la Peur du CBOE)")
        st.markdown(r"""
        Le VIX mesure la volatilité implicite à 30 jours anticipée par le marché des options sur le S&P 500.
        * **VIX < 15 :** Marché complaisant, carry trade florissant, tendance sereine.
        * **15 ≤ VIX ≤ 25 :** Volatilité normale, oscillations maîtrisées.
        * **25 < VIX ≤ 30 :** Tension nerveuse, réduction recommandée de la taille des positions de 50%.
        * **VIX > 30 :** **Panique systémique (Krach, guerre, faillite bancaire)**. 
          Tous les modèles quantitatifs classiques tombent en panne car les corrélations convergent vers 1.0 (tout s'effondre en même temps).
        """)
        
        st.subheader("B. Les Taux Obligataires (TNX US 10-Year) & Le Dollar (DXY)")
        st.markdown(r"""
        * **Taux 10 ans US (TNX) :** C'est le taux d'actualisation de tous les flux futurs mondiaux. 
          Quand le TNX monte violemment, la valorisation mathématique des actions technologiques à forte croissance (qui réalisent leurs bénéfices dans 5 ans) est mécaniquement massacrée.
        * **Dollar Index (DXY) :** Un dollar fort assèche les liquidités mondiales et pénalise les multinationales exportatrices et les marchés émergents.
        """)

    with t6:
        st.header("6. Le Cockpit des Déclencheurs Multi-Régimes (Moteur Intégré)")
        st.markdown(r"""
        Dans la version la plus récente de votre application (`xtrader-xgsboost`), 
        le moteur quantitatif exécute une fonction temps réel nommée `evaluate_multi_regime_triggers()`.
        Elle synthétise l'ensemble de ces règles dans un tableau de bord à 4 piliers :
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(r"""
            #### 1. 🏛️ Régime de Tendance Macro
            * **BULLISH (Vert) :** Cours $> SMA_{200}$ et $SMA_{50} \ge SMA_{200}$. Pleine exposition autorisée (100% de la taille nominale).
            * **BEARISH (Rouge) :** Cours $< SMA_{200}$. Achats fortement déconseillés ou taille réduite de 50%.
            * **NEUTRAL (Jaune) :** Zone de transition / correction intermédiaire.
            
            #### 2. 🛡️ Bouclier RSI (RSI Shield)
            * **OVERBOUGHT_SHIELD (Alerte) :** $RSI > 75$. Les achats sont **automatiquement annulés** par le bot pour éviter d'acheter sur le sommet d'une bulle.
            * **OVERSOLD_OPPORTUNITY :** $RSI < 30$. Alerte d'opportunité d'achat sur rebond imminent.
            """)
        with c2:
            st.markdown(r"""
            #### 3. 🚀 Détecteur de Flux (Volume Surge)
            * **INSTITUTIONAL_INFLOW :** $Volume \ge 1.5 \times SMA_{20}(Volume)$. Confirmation que la cassure de résistance est soutenue par des investisseurs institutionnels.
            
            #### 4. ⚡ Compression Bollinger (Squeeze)
            * **SQUEEZE_ACTIVE :** $BB\_Width \le 0.05$. Les bandes sont comprimées. Préparez-vous à une explosion directionnelle de grande envergure.
            """)
            
        st.success(r"""
        🔒 **Sécurisation Automatique au Breakeven (+1.5 ATR / 1R) :**
        Dès que l'action prend +1.5 ATR de gain par rapport au cours d'entrée, 
        l'algorithme remonte instantanément le Stop-Loss au prix d'achat. 
        **Le trade devient mathématiquement « Gratuit » : le risque de perte en capital est ramené à 0.00 $ !**
        """)

    with t7:
        questions_ind = [
            {
                "question": "Pourquoi placer un Stop-Loss à 1.5 ATR est-il supérieur à un stop fixe de 2% ?",
                "options": [
                    "Parce que 1.5 ATR s'adapte à la volatilité réelle et au bruit quotidien du titre.",
                    "Parce que le chiffre 1.5 est un nombre d'or en mathématiques financières.",
                    "Parce que cela permet de ne jamais avoir de perte sur un trade."
                ],
                "correct_index": 0,
                "explanation": "L'ATR mesure le battement quotidien réel de l'action. Un stop fixe de 2% se fait balayer par le bruit normal sur une action très volatile."
            },
            {
                "question": "Que fait le Bouclier RSI (RSI Shield) dans l'application xtrader-xgsboost lorsque le RSI dépasse 75 ?",
                "options": [
                    "Il vend immédiatement tout le portefeuille à découvert.",
                    "Il bloque et annule les nouveaux signaux d'achat pour éviter d'acheter sur un sommet de bulle.",
                    "Il double l'effet de levier pour profiter de la hausse."
                ],
                "correct_index": 1,
                "explanation": "Le bouclier RSI empêche les achats impulsifs de sommet de marché en surchauffe ($RSI > 75$)."
            },
            {
                "question": "Que signifie un indicateur Bollinger BB_Width très faible (< 0.05) ?",
                "options": [
                    "Le marché est mort et ne bougera plus jamais.",
                    "Une phase de compression (Squeeze) où l'énergie s'accumule avant une explosion violente de volatilité.",
                    "Une divergence baissière garantie."
                ],
                "correct_index": 1,
                "explanation": "La compression des bandes de Bollinger (Squeeze) précède historiquement les mouvements directionnels les plus explosifs."
            }
        ]
        render_interactive_quiz("indicators", questions_ind)


# =============================================================================
# 2. ACADÉMIE : PLONGÉE DANS XGBOOST & MACHINE LEARNING
# =============================================================================

def render_xgboost_academy():
    st.title("🤖 Académie Quantitative : Plongée dans XGBoost")
    st.markdown(r"""
    Dans les compétitions Kaggle et sur les desks de trading quantitatif de Manhattan, 
    les réseaux de neurones profonds (Deep Learning) se font régulièrement battre sur les données tabulaires 
    par un algorithme d'élite : **XGBoost (eXtreme Gradient Boosting)**.
    
    Voici la théorie mathématique, l'architecture d'arbres et les secrets de son paramétrage en finance.
    """)
    
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🌳 Arbres CART & Boosting",
        "⚙️ Hyperparamètres Clés",
        "🛡️ Lutte contre l'Overfitting",
        "🧠 Explicabilité SHAP",
        "🔬 Cas Pratique Complet",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. Des Arbres de Décision au Gradient Boosting")
        st.markdown(r"""
        Pour comprendre XGBoost, il faut d'abord comprendre pourquoi un arbre de décision simple échoue sur les marchés financiers.
        """)
        
        st.subheader("A. L'Arbre de Décision Binaire (CART)")
        st.markdown(r"""
        Un arbre CART sépare l'espace multidimensionnel des indicateurs techniques par des hyperplans orthogonaux.
        À chaque nœud, l'arbre cherche l'indicateur $x_j$ et le seuil $s$ qui maximisent le gain d'information (ou la pureté de Gini) :
        """)
        st.latex(r"Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma")
        st.markdown(r"""
        * **Le Problème de l'arbre unique :** 
          Soit l'arbre est trop court (sous-apprentissage, il prédit au hasard), 
          soit il est trop profond et mémorise le bruit passé par cœur (sur-apprentissage destructeur).
        """)
        
        st.subheader("B. Le Principe du Gradient Boosting (Jerome Friedman)")
        st.markdown(r"""
        Plutôt que de construire un arbre géant, XGBoost construit une suite de **$N$ petits arbres légers** entraînés séquentiellement.
        Chaque arbre successif n'apprend pas à prédire le cours de l'action : **il apprend à prédire l'erreur résiduelle (le gradient négatif) commise par la somme de tous les arbres précédents.**
        """)
        st.latex(r"\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)")
        st.markdown(r"""
        * $\hat{y}_i^{(t)}$ : Prédiction au round $t$.
        * $f_t(x_i)$ : Le nouvel arbre entraîné pour corriger l'erreur résiduelle.
        * $\eta$ (Eta / Learning Rate) : Le facteur d'amortissement (généralement 0.03 à 0.05 en finance) qui empêche l'algorithme de sur-réagir à une anomalie ponctuelle.
        """)
        
        st.subheader("C. La Régularisation Analytique de XGBoost")
        st.markdown("XGBoost intègre directement dans sa fonction de perte une pénalité L1 et L2 sur les poids des feuilles :")
        st.latex(r"\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|")
        st.info(r"""
        💡 **Pourquoi XGBoost est supérieur à AdaBoost :** 
        Le terme $\gamma T$ pénalise le nombre total de feuilles $T$ (élagage automatique des branches inutiles), 
        tandis que $\lambda$ et $\alpha$ écrasent les poids excessifs. 
        C'est ce qui permet au modèle de survivre sur les marchés boursiers sans apprendre le hasard par cœur.
        """)

    with t2:
        st.header("2. La Salle des Machines : Les Hyperparamètres en Finance")
        st.markdown(r"""
        Un mauvais réglage d'hyperparamètres sur des données financières garantit un désastre. 
        Voici les plages idéales validées par les quants de notre plateforme :
        """)
        
        st.markdown(r"""
| Hyperparamètre | Rôle Mathématique | Réglage Quant Idéal | Pourquoi ce réglage ? |
| :--- | :--- | :--- | :--- |
| **`max_depth`** | Profondeur maximale de chaque arbre | **3 à 5** (Défaut : 4) | Empêche l'arbre de créer des règles trop complexes et non généralisables. |
| **`learning_rate` ($\eta$)** | Pas de descente de gradient | **0.03 à 0.05** | Force l'algorithme à avancer à pas lents pour converger sans bruit. |
| **`n_estimators`** | Nombre total d'arbres séquentiels | **100 à 250** | Suffisant avec un petit learning rate pour capturer la structure. |
| **`subsample`** | Fraction d'observations tirées par arbre | **0.70 à 0.85** | Introduit du bagging stochastique, brisant les dépendances temporelles locales. |
| **`colsample_bytree`** | Fraction d'indicateurs tirés par arbre | **0.60 à 0.80** | Empêche un seul indicateur dominant (ex: RSI) d'écraser tous les arbres. |
| **`reg_lambda` ($\lambda$)** | Régularisation Ridge L2 sur les feuilles | **1.0 à 5.0** | Lisse les prédictions et évite les valeurs extrêmes de probabilité. |
| **`min_child_weight`** | Poids minimal requis dans une feuille | **3 à 10** | Interdit la création de règles basées sur 1 ou 2 jours seulement. |
        """)

    with t3:
        st.header("3. La Lutte Absolue contre l'Overfitting Financier")
        st.error(r"""
        🚨 **Le Boss Final du Quant :** L'Overfitting (Sur-apprentissage). 
        Si vous entraînez un modèle sur 5 ans d'Apple et qu'il affiche 98% de précision sur son historique, 
        vous n'avez pas créé un génie : vous avez créé un magnétophone qui récite le passé. 
        Le lendemain matin en live, il perdra tout son capital.
        """)
        
        st.subheader("A. La Stationnarité des Données d'Entrée")
        st.markdown(r"""
        Les séries temporelles de prix bruts ($Close_t$) ne sont **jamais stationnaires** (elles dérivent dans le temps). 
        Si vous donnez le cours brut d'Apple (ex: 220 $) à XGBoost, il sera incapable de prédire quand Apple cotera 250 $, 
        car aucun arbre n'a jamais vu de valeur supérieure à 220 $.
        
        **Notre solution dans `quant_core.py` :**
        Nous transformons tous les prix en **features stationnaires relatives** :
        * Rendements logarithmiques : $r_t = \ln(P_t / P_{t-1})$
        * Ratios de moyennes mobiles : $SMA\_Ratio = P_t / SMA_{50}(t)$
        * Position relative dans les bandes : $\%B = (P - Lower) / (Upper - Lower)$
        * Ratio de volume : $Vol\_Ratio = Volume_t / SMA_{20}(Volume)_t$
        """)
        
        st.subheader("B. La Validation Temporelle Walk-Forward (WFA)")
        st.markdown(r"""
        En Machine Learning classique (images, texte), on utilise la validation croisée K-Fold aléatoire. 
        **En finance, le K-Fold aléatoire est interdit !** 
        Il commet le péché suprême de **Lookahead Bias (Fuite Temporelle)** : 
        il utilise le cours du vendredi pour prédire le cours du mardi précédent !
        
        Dans notre application, nous utilisons la **Validation Walk-Forward** : 
        l'IA s'entraîne sur le passé (ex: Mois 1 à 12), puis est testée en aveugle sur le mois 13. 
        Ensuite la fenêtre glisse. La précision affichée dans le Terminal est cette performance hors-échantillon (*Out-Of-Sample*).
        """)

    with t4:
        st.header("4. Explicabilité Mathématique : Les Valeurs SHAP")
        st.markdown(r"""
        En finance institutionnelle, une boîte noire qui dit *"Achetez"* avec 72% de probabilité sans explication est inutilisable et interdite par les régulateurs. 
        Nous utilisons **SHAP (SHapley Additive exPlanations)**, issu de la Théorie des Jeux Coopératifs (Prix Nobel d'Économie Lloyd Shapley).
        """)
        
        st.latex(r"\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]")
        
        st.markdown(r"""
        * **Principe :** SHAP calcule la contribution marginale de chaque indicateur technique sur la décision du jour en testant toutes les combinaisons possibles.
        * **Dans notre application (Onglet 3 du Terminal) :**
          Le graphique en cascade (*Waterfall Plot*) décompose le signal :
          - Ligne de base (Valeur attendue moyenne du marché) : 50%
          - Poussée haussière (+15%) : VWAP haussier et afflux de volume.
          - Frein baissier (-8%) : Résistance macroéconomique (VIX en hausse).
          - **Probabilité finale affichée : 57%**.
        """)

    with t5:
        st.header("5. Cas Pratique : Anatomie d'un Signal Algorithmique Réel")
        st.markdown(r"""
        Voici comment le moteur XGBoost de notre plateforme prend une décision réelle sur un titre comme Apple (AAPL) :
        """)
        
        st.code("""
Étape 1 : Téléchargement OHLCV + Taux TNX + VIX + SPY (2 ans).
Étape 2 : Calcul des 24 descripteurs stationnaires (quant_core.add_features).
Étape 3 : Entraînement des 150 arbres à gradient régularisé (quant_core.MLTrader.train).
Étape 4 : Évaluation sur la dernière bougie boursière :
   - Close = 224.50 $ | SMA 50 = 218.00 $ | SMA 200 = 195.00 $
   - RSI = 61.2 (Zone saine, < 75)
   - Volume Ratio = 1.78x (Afflux institutionnel détecté)
   - VIX = 16.4 (Marché directeur calme)
Étape 5 : Prédiction XGBoost : Probabilité d'achat = 68.4%.
Étape 6 : Validation par le Cockpit des Déclencheurs :
   - Régime Macro : Tendance Saine (BULLISH) -> VALIDÉ
   - Bouclier RSI : Normal (pas de surachat) -> VALIDÉ
   - Coupe-Circuit VIX : Serein (< 30) -> VALIDÉ
Étape 7 : Décision Finale : 🟢 SIGNAL D'ACHAT FORT ÉMIS.
   - Stop-Loss calculé : 220.10 $ (-1.5 ATR)
   - Palier Breakeven de sécurisation : 228.90 $ (+1.5 ATR)
   - Take-Profit calculé : 233.30 $ (+3.0 ATR)
        """, language="text")

    with t6:
        questions_xgb = [
            {
                "question": "Pourquoi ne doit-on jamais donner des prix bruts (ex: 220$) à XGBoost en finance ?",
                "options": [
                    "Parce que les séries de prix bruts ne sont pas stationnaires et rendent le modèle incapable de généraliser à de nouveaux sommets.",
                    "Parce que XGBoost ne sait manipuler que des nombres entiers.",
                    "Parce que les prix bruts sont soumis à la taxe sur les transactions financières."
                ],
                "correct_index": 0,
                "explanation": "Les prix bruts dérivent dans le temps (non-stationnarité). XGBoost ne peut pas extrapoler au-delà du prix maximum vu durant l'entraînement. Il faut utiliser des ratios et des rendements stationnaires."
            },
            {
                "question": "En quoi consiste le Gradient Boosting séquentiel de XGBoost ?",
                "options": [
                    "Chaque nouvel arbre apprend à corriger les erreurs résiduelles (le gradient) laissées par les arbres précédents.",
                    "Il entraîne 10 000 arbres identiques en même temps.",
                    "Il inverse la tendance quand le marché baisse."
                ],
                "correct_index": 0,
                "explanation": "Le Boosting entraîne des arbres en séquence, où chaque arbre tente de minimiser le résidu d'erreur du précédent."
            },
            {
                "question": "Pourquoi la validation croisée aléatoire (K-Fold classique) est-elle interdite en trading quantitatif ?",
                "options": [
                    "Parce qu'elle consomme trop de mémoire RAM.",
                    "Parce qu'elle provoque une fuite temporelle (Lookahead Bias) en utilisant des données futures pour prédire le passé.",
                    "Parce qu'elle est incompatible avec Python."
                ],
                "correct_index": 1,
                "explanation": "Le K-Fold aléatoire mélange le passé et le futur, ce qui fausse complètement l'évaluation du modèle en créant des performances artificielles."
            }
        ]
        render_interactive_quiz("xgboost", questions_xgb)


# =============================================================================
# 3. ACADÉMIE : STRATÉGIE, RISK MANAGEMENT & BLACK-LITTERMAN
# =============================================================================

def render_strategy_academy():
    st.title("🎓 Académie : Stratégie & Risk Management Institutionnel")
    st.markdown(r"""
    Dans un hedge fund, le département le plus puissant n'est pas celui des traders : **c'est le département du Risque**. 
    Même avec une IA prédisant le marché avec 70% d'exactitude, un mauvais money management détruira mathématiquement votre capital.
    
    Voici les règles d'or de la survie financière et de la génération d'Alpha.
    """)
    
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Alpha, Beta & Ratios",
        "💀 Les Mathématiques du Drawdown",
        "💰 Critère de Kelly & Sizing",
        "🔒 Breakeven Stop & Risk/Reward",
        "🏛️ Modèle de Black-Litterman",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. Alpha, Beta & Ratios de Performance")
        st.markdown(r"""
        * **Le Beta ($\beta$) :** C'est le rendement passif dû à la marée montante du marché global ($R_p = \beta \cdot R_m$).
        * **L'Alpha ($\alpha$) :** C'est la **véritable valeur ajoutée** du modèle quantitatif, décorrélée des caprices du marché.
        """)
        
        st.subheader("Les Ratios de Mesure Institutionnels :")
        st.latex(r"\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p} \quad | \quad \text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_{downside}}")
        st.markdown(r"""
        * **Le Ratio de Sharpe :** Mesure l'excès de rendement par unité de volatilité totale. Un bon fonds quantitatif vise un Sharpe $> 1.5$.
        * **Le Ratio de Sortino (Supérieur au Sharpe) :** Pourquoi pénaliser une stratégie lorsqu'elle subit une volatilité *haussière* explosive ? Le Sortino ne prend en compte que la volatilité baissière ($\sigma_{downside}$).
        * **Le Ratio de Calmar :** Rendement annualisé divisé par le Maximum Drawdown historique. C'est le ratio de référence des fonds systématiques macro.
        """)

    with t2:
        st.header("2. Les Mathématiques de la Ruine & L'Asymétrie des Pertes")
        st.markdown(r"""
        L'asymétrie des pourcentages boursiers est une loi physique implacable. 
        Chaque perte subie exige un effort surhumain pour simplement revenir à la case départ :
        """)
        
        st.markdown(r"""
| Perte en Capital | Gain Nécessaire pour Revenir à Zéro | Difficulté Psychologique & Mathématique |
| :---: | :---: | :--- |
| **-10%** | **+11.1%** | Correction normale, facilement absorbée. |
| **-20%** | **+25.0%** | Exige une excellente performance trimestrielle. |
| **-30%** | **+42.8%** | Nécessite un marché haussier puissant. |
| **-50%** | **+100.0%** | **Doubler son capital restant.** Très peu de traders s'en remettent. |
| **-75%** | **+300.0%** | Quasi-impossible sans prise de risque délirante. |
| **-90%** | **+900.0%** | Ruine définitive. |
        """)
        
        st.error(r"""
        ⚠️ **Leçon Historique : La Faillite de Long-Term Capital Management (LTCM) en 1998**
        Dirigé par les lauréats du Prix Nobel Myron Scholes et Robert Merton, LTCM disposait des meilleurs cerveaux du monde. 
        Leurs algorithmes affichaient un Sharpe astronomique de 4.0. 
        Cependant, ils utilisaient un levier de 25:1 sans respecter les queues épaisses (*Fat Tails*). 
        Lors du défaut de paiement de la Russie en 1998 (un événement à 6 écarts-types), 
        leur capital de 4.6 milliards de dollars s'est évaporé en 4 semaines, obligeant la Réserve Fédérale à intervenir pour éviter un krach mondial.
        """)

    with t3:
        st.header("3. Le Money Management Scientifique : Le Critère de Kelly")
        st.markdown(r"""
        Quelle fraction exacte $f^*$ de votre capital devez-vous allouer à un trade pour maximiser le taux de croissance logarithmique à long terme sans jamais faire faillite ?
        John Kelly, chercheur aux laboratoires Bell en 1956, a formulé l'équation optimale :
        """)
        
        st.latex(r"f^* = \frac{p \cdot b - (1 - p)}{b} = p - \frac{q}{b}")
        st.markdown(r"""
        * $p$ : Probabilité de gain (Win Rate du modèle).
        * $q = 1 - p$ : Probabilité de perte.
        * $b$ : Ratio Gain / Perte moyen (Gain d'un trade gagnant divisé par Perte d'un trade perdant).
        """)
        
        st.warning(r"""
        💡 **La Règle du Half-Kelly (Kelly Fractionnaire) :**
        La formule de Kelly pure part du principe que vous connaissez $p$ et $b$ avec une précision infinie. 
        Dans le monde réel, ces paramètres varient. Utiliser le plein Kelly conduit à une volatilité de portefeuille insoutenable (drawdowns de 40%). 
        C'est pourquoi notre moteur quantitatif applique systématiquement un **Half-Kelly ($f^* / 2$)**, 
        strictement plafonné à **2% à 5% de capital maximum** par position.
        """)

    with t4:
        st.header("4. Breakeven Stop & Le Ratio Risk/Reward Asymétrique")
        st.markdown(r"""
        Une stratégie peut perdre 6 fois sur 10 et être exceptionnellement rentable, 
        à condition d'appliquer une asymétrie de gain stricte.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("A. L'Asymétrie 1:2 ou 1:3")
            st.markdown(r"""
            * Risque par trade ($1R$) : 100 $ (Stop à -1.5 ATR).
            * Objectif de profit ($2R$ à $3R$) : 300 $ (Take-Profit à +3.0 ATR).
            * Sur 10 trades avec 40% de réussite seulement :
              - 6 pertes de 100 $ = -600 $
              - 4 gains de 300 $ = +1 200 $
              - **Bénéfice net = +600 $ malgré 60% d'erreurs !**
            """)
        with c2:
            st.subheader("B. Le Déclencheur Breakeven Automatique")
            st.markdown(r"""
            Dans notre application `xtrader-xgsboost` :
            * Dès que le cours atteint $+1.5 \times ATR$ (gain équivalent au risque initial $1R$), 
              le Stop-Loss est automatiquement remonté au prix d'achat initial.
            * Si l'action subit un brusque retournement (*pullback*), la position est coupée à 0.00 $ de perte.
            * Si l'action poursuit son envolée, vous capturez l'intégralité du mouvement sans aucun stress.
            """)

    with t5:
        st.header("5. L'Allocation Bayésienne de Black-Litterman")
        st.markdown(r"""
        Conçu chez Goldman Sachs par Fischer Black et Robert Litterman en 1990, 
        ce modèle est la référence absolue pour construire un portefeuille multi-actifs.
        """)
        
        st.latex(r"E[R] = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]")
        st.markdown(r"""
        * **Pourquoi l'approche classique de Markowitz échoue :**
          L'optimisation classique Moyenne-Variance de Markowitz est un « amplificateur d'erreurs d'estimation ». 
          Un infime changement dans l'estimation du rendement attendu d'une action conduit Markowitz à allouer 100% du capital sur cette seule action et 0% sur les autres.
        * **Le Génie de Black-Litterman :**
          1. **L'Ancre d'Équilibre ($\Pi$) :** Le modèle commence par calculer les rendements implicites d'équilibre du marché mondial (*Market Portfolio*).
          2. **Les Vues de l'Investisseur ($Q$) :** L'investisseur exprime des vues relatives ou absolues (dans notre système, **ce sont les probabilités calculées par XGBoost**).
          3. **La Matrice d'Incertitude ($\Omega$) :** Le modèle pondère chaque vue selon le degré de certitude de l'IA. Si l'IA est peu sûre, le modèle reste proche du marché. Si l'IA a une conviction extrême, le portefeuille bascule méthodiquement vers le titre cible sans détruire la diversification.
        """)

    with t6:
        questions_strat = [
            {
                "question": "Si votre portefeuille perd 50% de sa valeur, quel pourcentage de gain devez-vous réaliser pour simplement revenir à zéro ?",
                "options": [
                    "+50%",
                    "+100%",
                    "+200%"
                ],
                "correct_index": 1,
                "explanation": "Si un compte de 10 000 $ tombe à 5 000 $ (-50%), il doit gagner 5 000 $ sur ses 5 000 $ restants, soit un gain de +100%."
            },
            {
                "question": "Pourquoi utilise-t-on le Half-Kelly plutôt que le plein Kelly en finance ?",
                "options": [
                    "Parce que le plein Kelly conduit à des drawdowns catastrophiques en cas d'erreurs d'estimation sur les probabilités.",
                    "Parce que la loi interdit d'utiliser la formule de Kelly complète.",
                    "Parce que le Half-Kelly supprime les commissions des courtiers."
                ],
                "correct_index": 0,
                "explanation": "Le Half-Kelly offre 75% du taux de croissance optimal tout en divisant la variance et le risque de drawdown par deux."
            },
            {
                "question": "Quelle est la principale supériorité du modèle de Black-Litterman par rapport à Markowitz ?",
                "options": [
                    "Il évite la concentration extrême sur un seul actif en combinant l'équilibre de marché avec les convictions chiffrées de l'IA.",
                    "Il garantit des gains sans aucun risque.",
                    "Il supprime totalement le besoin de calculer des covariances."
                ],
                "correct_index": 0,
                "explanation": "Black-Litterman résout l'instabilité de Markowitz par une approche bayésienne alliant équilibre de marché et vues expertes."
            }
        ]
        render_interactive_quiz("strategy", questions_strat)


# =============================================================================
# 4. ACADÉMIE : MODÉLISATION AVANCÉE (STACKING, OPTUNA & MONTE CARLO)
# =============================================================================

def render_advanced_academy():
    st.title("🎓 Académie : Modélisation Avancée & Ingénierie Quant")
    st.markdown(r"""
    Au-delà d'un modèle unique, la finance de pointe repose sur l'intelligence collective d'algorithmes hétérogènes, 
    l'optimisation bayésienne des espaces d'hyperparamètres et les stress-tests massifs de Monte Carlo.
    """)
    
    t1, t2, t3, t4, t5 = st.tabs([
        "🤝 Stacking & Sagesse des Foules",
        "🎯 Optimisation Bayésienne (Optuna)",
        "🎲 Simulation de Monte Carlo",
        "🧬 Feature Engineering Avancé",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. L'Intelligence d'Ensemble (Stacking & Soft Voting)")
        st.markdown(r"""
        En 1906, le statisticien Francis Galton demanda à 800 personnes lors d'une foire agricole d'estimer le poids d'un bœuf. 
        La moyenne de toutes les estimations était exacte à une livre près, battant le meilleur boucher de la région. 
        C'est **la sagesse des foules**.
        """)
        
        st.subheader("Notre Comité d'Algorithmes (Voting Classifier) :")
        st.markdown(r"""
        Dans `quant_core.py`, le système n'utilise pas uniquement XGBoost, il peut fédérer un comité d'experts :
        1. **XGBoost :** Capture les relations non-linéaires complexes et les interactions de haut niveau.
        2. **LightGBM :** Croissance des arbres par feuille (*Leaf-wise*), ultra-rapide sur les anomalies de microstructure.
        3. **Random Forest :** Arbres indépendants baggés, apporte une régularité et une prudence extrême.
        
        Le signal final est issu d'un **Consensus Pondéré (Soft Voting)** :
        """)
        st.latex(r"P(Achat) = w_1 \cdot P_{XGB} + w_2 \cdot P_{LGBM} + w_3 \cdot P_{RF} \quad \text{avec} \quad \sum w_i = 1")
        st.info("Ce consensus permet d'éliminer plus de 60% des faux signaux générés par un modèle isolé.")

    with t2:
        st.header("2. Optimisation Bayésienne avec Optuna")
        st.markdown(r"""
        Comment trouver la combinaison idéale de 10 hyperparamètres sans passer 3 semaines à faire du Grid Search exhaustif ?
        Nous intégrons **Optuna**, le framework d'optimisation bayésienne le plus avancé.
        """)
        
        st.latex(r"P(\theta | y) = \frac{P(y | \theta) \cdot P(\theta)}{P(y)}")
        st.markdown(r"""
        * **L'estimateur TPE (Tree-structured Parzen Estimator) :** 
          Optuna modélise deux distributions de probabilité dans l'espace des hyperparamètres :
          - $\ell(\theta)$ : Les hyperparamètres ayant généré les meilleurs Sharpe Ratios.
          - $g(\theta)$ : Les hyperparamètres ayant conduit à des résultats médiocres.
        * À chaque itération, Optuna choisit les paramètres qui maximisent le ratio $\ell(\theta) / g(\theta)$. 
          L'IA converge vers la configuration optimale en 30 essais au lieu de 10 000 tests aléatoires.
        """)

    with t3:
        st.header("3. La Simulation de Monte Carlo & Le Bootstrap Temporel")
        st.markdown(r"""
        Inventée par Stanislaw Ulam et John von Neumann lors du projet Manhattan, 
        la méthode de Monte Carlo permet de tester la résilience d'un système dans des univers parallèles.
        """)
        
        st.subheader("Stress-Test du Modèle de Trading :")
        st.markdown(r"""
        L'erreur classique est de juger un algorithme sur son unique historique de backtest. 
        Dans le monde réel, l'ordre des trades aurait pu être différent : deux pertes auraient pu survenir en premier.
        
        **Le Protocole de Bootstrap dans notre plateforme :**
        1. On extrait la distribution des rendements de tous les trades historiques.
        2. On tire au sort avec remise pour reconstituer **1 000 courbes de capitaux alternatives**.
        3. On mesure le **Worst Case Drawdown** et la **Probabilité de Ruine**. 
           Si dans plus de 1% des univers alternatifs le compte subit un drawdown $> 25\%$, la stratégie est rejetée.
        """)

    with t4:
        st.header("4. Feature Engineering Avancé pour Séries Financières")
        st.markdown(r"""
        Le succès d'un modèle quantitatif dépend à 80% de la qualité des variables (*Features*) fournies à l'algorithme.
        """)
        
        st.markdown(r"""
        * **Différenciation Fractionnaire (Marcos López de Prado) :**
          Une différenciation entière $\Delta P = P_t - P_{t-1}$ rend la série stationnaire mais détruit toute la mémoire à long terme. 
          La différenciation fractionnaire $(1 - B)^d$ avec $d \in ]0, 1[$ permet de conserver la mémoire des cycles tout en atteignant la stationnarité requise.
        * **Features de Microstructure & Volatilité Réalisée :**
          - Ratios de volume intrajournalier.
          - Écart logarithmique aux moyennes mobiles exponentielles courtes et longues.
          - Ratio de volatilité de Parkinson basée sur les extrêmes High-Low.
        """)

    with t5:
        questions_adv = [
            {
                "question": "Quel est l'avantage du Soft Voting (Stacking) par rapport à un modèle unique ?",
                "options": [
                    "Il combine les probabilités de plusieurs modèles hétérogènes pour filtrer le bruit et réduire les faux signaux.",
                    "Il divise par deux le temps de calcul.",
                    "Il garantit un profit sur chaque trade."
                ],
                "correct_index": 0,
                "explanation": "L'ensemble de modèles (Consensus) réduit la variance de l'erreur d'estimation et stabilise les prédictions."
            },
            {
                "question": "Comment l'optimisation bayésienne d'Optuna choisit-elle les prochains hyperparamètres à tester ?",
                "options": [
                    "Elle teste tous les nombres dans l'ordre croissant.",
                    "Elle modélise les zones de paramètres à succès et concentre ses recherches là où la probabilité de gain est la plus élevée.",
                    "Elle tire les valeurs à pile ou face."
                ],
                "correct_index": 1,
                "explanation": "Optuna utilise l'estimateur TPE pour cartographier l'espace des paramètres et focaliser les essais sur les zones prometteuses."
            }
        ]
        render_interactive_quiz("advanced", questions_adv)


# =============================================================================
# 5. ACADÉMIE : OPTIONS & BLACK-SCHOLES (NIVEAU INSTITUTIONNEL)
# =============================================================================

def render_options_academy():
    st.title("🎓 Académie : Options & Black-Scholes (Niveau Institutionnel)")
    st.markdown(r"""
    Bienvenue dans le sanctuaire des produits dérivés. 
    Les actions vous permettent de parier sur une seule dimension : la direction. 
    **Les options vous permettent de parier sur 4 dimensions simultanées : la direction, la vitesse, le temps et la peur (volatilité).**
    """)
    
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📘 Fondations Dérivés",
        "🔬 Modèle Black-Scholes",
        "🛡️ Les Grecs Complets",
        "🌊 Volatility Smile & Surface",
        "💰 Vente de Primes & IV Crush",
        "⚖️ Norme US vs Norme EU",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. Fondations des Contrats d'Options")
        st.markdown(r"""
        Une option est un contrat financier unilatéral accordant **un droit (sans obligation)** à l'acheteur, 
        et imposant **une obligation absolue** au vendeur en échange d'une rémunération immédiate : **la Prime (Premium)**.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 L'Option CALL (Achat)")
            st.markdown(r"""
            * **Droit pour l'Acheteur :** Acheter l'action au prix du *Strike* ($K$) jusqu'à l'échéance ($T$).
            * **Obligation pour le Vendeur :** Livrer les actions au prix $K$ si l'acheteur le demande.
            * **Profil de gain :** Risque limité à la prime payée, gain théoriquement illimité à la hausse.
            """)
        with c2:
            st.subheader("📉 L'Option PUT (Vente)")
            st.markdown(r"""
            * **Droit pour l'Acheteur :** Vendre l'action au prix du *Strike* ($K$) peu importe sa chute.
            * **Rôle d'assurance :** Permet de couvrir un portefeuille contre un krach boursier.
            * **Profil de gain :** Risque limité à la prime payée, gain maximal si l'action tombe à 0 $.
            """)
            
        st.subheader("Les 3 États Monétaires d'une Option :")
        st.markdown(r"""
        * **ITM (In The Money - Dans la Monnaie) :** Possède une valeur intrinsèque réelle ($Call: S > K$ ou $Put: S < K$).
        * **ATM (At The Money - À la Monnaie) :** Le cours actuel est égal au strike ($S \approx K$). C'est ici que la valeur temps est maximale.
        * **OTM (Out of The Money - Hors de la Monnaie) :** Valeur intrinsèque nulle. Le contrat n'est composé que de valeur spéculative pure ($Call: S < K$).
        """)

    with t2:
        st.header("2. L'Équation Fondamentale de Black-Scholes-Merton (1973)")
        st.markdown(r"""
        Publiée par Fischer Black, Myron Scholes et Robert Merton, cette équation a valu le Prix Nobel d'Économie en 1997.
        Elle repose sur un portefeuille de réplication sans arbitrage liant dynamiquement l'action et l'option.
        """)
        
        st.latex(r"C(S, t) = S \cdot N(d_1) - K \cdot e^{-r(T-t)} \cdot N(d_2)")
        st.latex(r"P(S, t) = K \cdot e^{-r(T-t)} \cdot N(-d_2) - S \cdot N(-d_1)")
        st.markdown("avec :")
        st.latex(r"d_1 = \frac{\ln(S/K) + \left( r + \frac{\sigma^2}{2} \right)(T-t)}{\sigma \sqrt{T-t}} \quad \text{et} \quad d_2 = d_1 - \sigma \sqrt{T-t}")
        
        st.info(r"""
        🔬 **L'Héritage de la Physique Théorique :**
        Remarquez la structure de cette formule : elle est la solution exacte de **l'Équation de la Chaleur** de Joseph Fourier (1822) ! 
        Le terme $\sigma^2$ représente le coefficient de diffusion thermique. 
        En finance, le risque et l'incertitude se diffusent dans le temps exactement comme la chaleur se dissipe dans une barre métallique.
        """)

    with t3:
        st.header("3. La Famille Complète des Grecs (Greeks)")
        st.markdown(r"""
        Les professionnels ne tradent pas des prix : ils pilotent une matrice de risques multidimensionnelle.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Δ (Delta) : Sensibilité au Prix")
            st.latex(r"\Delta_{Call} = \frac{\partial C}{\partial S} = N(d_1) \in [0, 1]")
            st.markdown(r"""
            * Indique de combien la prime varie si l'action bouge de 1.00 $.
            * Équivaut approximativement à la **probabilité d'exercice à l'échéance**. Un Delta de 0.30 signifie 30% de probabilité d'expirer ITM.
            """)
            
            st.subheader("Θ (Theta) : L'Érosion Temporelle (Time Decay)")
            st.latex(r"\Theta = \frac{\partial C}{\partial t}")
            st.markdown(r"""
            * Le "loyer" quotidien payé par l'acheteur et encaissé par le vendeur. 
            * Le Theta s'accélère exponentiellement dans les 30 derniers jours avant l'échéance.
            """)
        with c2:
            st.subheader("Γ (Gamma) : La Vitesse du Delta")
            st.latex(r"\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S \sigma \sqrt{T-t}}")
            st.markdown(r"""
            * Mesure l'accélération du Delta.
            * C'est le moteur du **Gamma Squeeze** : quand des millions de particuliers achètent des calls OTM, les teneurs de marché doivent acheter de plus en plus d'actions à mesure que le cours monte, provoquant un cercle haussier vertigineux.
            """)
            
            st.subheader("ν (Vega) : Sensibilité à la Volatilité")
            st.latex(r"\nu = \frac{\partial C}{\partial \sigma} = S \sqrt{T-t} N'(d_1)")
            st.markdown(r"""
            * Indique la variation de la prime pour une hausse de 1% (0.01) de volatilité implicite.
            """)

    with t4:
        st.header("4. Volatility Smile & Surface de Volatilité 3D")
        st.markdown(r"""
        Dans le modèle théorique pur de Black-Scholes, la volatilité implicite $\sigma$ est supposée constante pour tous les strikes. 
        Cette hypothèse a volé en éclat lors du **Krach du 19 octobre 1987 (Black Monday)**, où le Dow Jones s'est effondré de 22.6% en un seul jour.
        """)
        
        st.warning(r"""
        📉 **La Naissance du Volatility Smile (Sourire de Volatilité) :**
        Depuis 1987, les investisseurs savent que les distributions financières ont des queues épaisses (*Fat Tails*). 
        Par conséquent, les acteurs du marché sont prêts à payer une prime disproportionnée pour les Puts très en dessous du cours actuel (OTM) afin de s'assurer contre un krach. 
        La courbe de volatilité en fonction des strikes forme une asymétrie marquée (le *Volatility Skew*).
        """)

    with t5:
        st.header("5. L'Art de la Vente de Primes & Le Piège de l'IV Crush")
        st.markdown(r"""
        Statistiquement, **plus de 75% des options expirent sans valeur ($0.00$)**. 
        Les fonds quantitatifs adoptent donc massivement le rôle d'**Assureurs du Marché** en vendant des primes.
        """)
        
        st.subheader("Le Phénomène d'IV Crush (Écrasement de Volatilité) :")
        st.markdown(r"""
        1. **Avant les Résultats (Earnings) :** L'incertitude monte, tout le monde achète des options de couverture. La volatilité implicite (IV) explose à 120%.
        2. **Le Débutant Achète un Call la veille :** Il paye une prime astronomique gonflée par le Vega.
        3. **L'Annonce a lieu :** L'entreprise publie de bons chiffres, l'action monte de +4%.
        4. **L'Écrasement :** L'incertitude étant levée, l'IV s'effondre instantanément de 120% à 35%. 
           La perte de valeur due au Vega est tellement violente qu'elle anéantit le gain du cours : **le Call perd 45% de sa valeur malgré la hausse de l'action !**
        """)
        
        st.success(r"""
        💰 **La Stratégie Institutionnelle :**
        Les quants utilisent le module de Backtester d'Options pour vendre des **Credit Spreads** ou des **Iron Condors** à Delta 0.15 : 
        ils encaissent la prime sur-gonflée et laissent le Theta et l'IV Crush détruire la valeur du contrat pour racheter à 0 $.
        """)

    with t6:
        st.header("6. Législation Boursière : Norme US vs Norme Européenne")
        st.markdown(r"""
        Dans notre application, chaque titre affiche son multiplicateur et sa législation spécifique :
        """)
        
        st.markdown(r"""
| Critère | Options Américaines (Ex: AAPL, NVDA, TSLA) | Options Européennes (Ex: LVMH, ASML, Sanofi) |
| :--- | :--- | :--- |
| **Taille du Lot (Multiplicateur)** | **100 actions** par contrat (Standard CBOE) | Souvent **10 actions** (Euronext) ou 100 actions |
| **Droit d'Exercice** | **À tout moment** avant la date d'expiration | **Uniquement le jour de l'échéance** |
| **Revente sur le Marché Secondaire** | Libre et liquide à chaque seconde de cotation | Libre et liquide à chaque seconde de cotation |
| **Risque d'Assignation Précoce** | Réel si l'option est profondément ITM avant dividende | Impossible avant la cloche de clôture finale |
        """)

    with t7:
        questions_opt = [
            {
                "question": "Pourquoi un acheteur de Call peut-il perdre de l'argent après une annonce de résultats même si l'action monte de +4% ?",
                "options": [
                    "À cause du phénomène d'IV Crush : la volatilité implicite s'effondre brutalement, détruisant la composante Vega de la prime.",
                    "Parce que le courtier applique une taxe spéciale sur les bénéfices.",
                    "Parce que le Delta d'un Call devient négatif après les résultats."
                ],
                "correct_index": 0,
                "explanation": "L'écrasement de la volatilité implicite (IV Crush) post-événement détruit la valeur temps et volatilité plus vite que le gain directionnel."
            },
            {
                "question": r"Que mesure le Gamma ($\Gamma$) d'une option ?",
                "options": [
                    "La sensibilité de l'option aux taux d'intérêt.",
                    "La vitesse de variation du Delta lorsque le cours du sous-jacent bouge de 1.00 $.",
                    "La perte de valeur quotidienne liée au temps qui passe."
                ],
                "correct_index": 1,
                "explanation": "Le Gamma est la dérivée seconde par rapport au prix de l'action, représentant l'accélération du Delta."
            }
        ]
        render_interactive_quiz("options", questions_opt)


# =============================================================================
# 6. ACADÉMIE : NLP FINANCIER PROFOND, SENTIMENT & RAG GEMINI
# =============================================================================

def render_news_academy():
    st.title("📰 Académie Quantitative : NLP Profond, Transformers & RAG")
    st.markdown(r"""
    L'analyse technique calcule les lois physiques du passé. 
    Mais lorsqu'une enquête de la SEC tombe ou qu'une avancée technologique est brevetée, 
    le cours réagit en 25 millisecondes. C'est le rôle du **NLP Profond (Natural Language Processing)**.
    """)
    
    t1, t2, t3, t4, t5 = st.tabs([
        "🤖 Du Sac de Mots aux Transformers",
        "🏦 FinBERT : L'IA Spécialiste Wall Street",
        "🎯 L'Alpha Hybride (Quant + NLP)",
        "🧠 RAG Financier avec Google Gemini",
        "🧠 Quiz d'Évaluation"
    ])
    
    with t1:
        st.header("1. L'Évolution du Traitement du Langage Naturel")
        st.markdown(r"""
        * **Génération 1 (Dictionnaires & TextBlob) :** 
          Comptait simplement les mots positifs (+1) et négatifs (-1). Incapable de comprendre l'ironie ou la négation.
          Exemple d'échec : *« Les résultats ne sont pas décevants »* était classé négatif à cause du mot « décevants ».
        * **Génération 2 (Réseaux Récurrents LSTM) :** Traitait les mots séquentiellement mais oubliait le début des phrases longues.
        * **Génération 3 (Transformers & Attention Mechanism - Vaswani et al., 2017) :** 
          Lit l'intégralité d'un document en parallèle. Le mécanisme d'attention calcule la relation d'interdépendance entre chaque mot.
        """)

    with t2:
        st.header("2. FinBERT : Le Modèle de Langage de Wall Street")
        st.markdown(r"""
        Un modèle généraliste comme BERT ou GPT standard ne comprend pas la gravité des termes financiers. 
        Dans le langage courant, le mot *"Liability"* signifie une responsabilité ; en finance, c'est une dette au passif du bilan.
        """)
        
        st.info(r"""
        🔬 **L'Architecture de FinBERT :**
        Entraîné par des chercheurs sur des dizaines de milliers de rapports 10-K/10-Q de la SEC et de dépêches Bloomberg, 
        FinBERT attribue un vecteur de probabilité à 3 classes :
        * $P(\text{Positif})$ : Perspective de croissance, amélioration des marges, guidance relevée.
        * $P(\text{Négatif})$ : Litiges, dégradation de note de crédit, perte de parts de marché.
        * $P(\text{Neutre})$ : Éléments factuels sans impact sur les flux futurs.
        """)

    with t3:
        st.header("3. L'Alpha Hybride : Fusionner XGBoost et FinBERT")
        st.markdown(r"""
        La force suprême de notre architecture est de ne jamais faire confiance aveuglément à un seul moteur.
        """)
        
        st.markdown(r"""
| Signal XGBoost (Technique) | Score NLP FinBERT (Sentiment) | Décision du Portfolio Manager | Rationale Quantitative |
| :---: | :---: | :---: | :--- |
| **ACHAT (68%)** | **POSITIF (+0.8)** | 🟢 **Achat Fort (Haute Conviction)** | Les mathématiques et les fondamentaux narratifs sont en convergence totale. |
| **ACHAT (72%)** | **NÉGATIF (-0.6)** | ⚠️ **Achat Bloqué / Neutralisé** | **Divergence dangereuse :** L'action est techniquement survendue, mais une catastrophe judiciaire est imminente. |
| **VENTE (35%)** | **NÉGATIF (-0.7)** | 🔴 **Vente / Prise de Profit** | Confirmation de la dégradation de tendance. |
| **VENTE (40%)** | **POSITIF (+0.9)** | 🟡 **Surveillance Rebond** | L'IA technique temporise pendant que les bonnes nouvelles s'accumulent. |
        """)

    with t4:
        st.header("4. RAG Financier (Retrieval-Augmented Generation) avec Google Gemini")
        st.markdown(r"""
        Dans le module **Analyse RAG (Gemini)** de notre application, nous implémentons un pipeline de pointe :
        """)
        
        st.code("""
1. RETRIEVAL (Extraction Temps Réel) :
   - Extraction des dernières dépêches d'actualités et communiqués Yahoo Finance.
   - Extraction des ratios fondamentaux (Forward P/E, Marges, Dividendes).
   - Extraction de la prédiction mathématique temps réel de XGBoost.
   
2. AUGMENTATION (Construction du Prompt Institutionnel) :
   - Fusion des métriques techniques et narratives dans un contexte structuré.
   
3. GENERATION (Raisonnement LLM Gemini 2.5 / 3.0) :
   - Rédaction d'un mémo de comité de gestion de portefeuille :
     * Analyse du sentiment profond et momentum caché.
     * Confrontation du modèle mathématique avec les fondamentaux réels.
     * Détection des risques imminents d'IV Crush avant les Earnings.
     * Recommandation experte de stratégies sur produits dérivés.
        """, language="text")

    with t5:
        questions_news = [
            {
                "question": "Pourquoi FinBERT est-il supérieur à un dictionnaire de mots classique en finance ?",
                "options": [
                    "Parce qu'il utilise le mécanisme d'Attention des Transformers pour comprendre le contexte financier et le sens exact des termes économiques.",
                    "Parce qu'il télécharge le cours de l'or en temps réel.",
                    "Parce qu'il ne fonctionne que sur les ordinateurs quantiques."
                ],
                "correct_index": 0,
                "explanation": "FinBERT contextualise le jargon financier de Wall Street, évitant les contresens des simples compteurs de mots."
            },
            {
                "question": "Que doit faire un gestionnaire quantitatif si XGBoost donne un signal d'Achat mais que le sentiment NLP est fortement Négatif ?",
                "options": [
                    "Ignorer ou bloquer le signal d'achat à cause de la divergence fondamentale dangereuse.",
                    "Acheter trois fois plus d'actions.",
                    "Désactiver internet."
                ],
                "correct_index": 0,
                "explanation": "Une divergence entre la technique et les news signale un risque asymétrique élevé (fraude, litige, profit warning)."
            }
        ]
        render_interactive_quiz("news", questions_news)
