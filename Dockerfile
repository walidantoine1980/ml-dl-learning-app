FROM python:3.11-slim

WORKDIR /app

# Définition des variables d'environnement
ENV HF_HOME=/app/.cache
ENV NLTK_DATA=/usr/share/nltk_data
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système (y compris libgomp1 obligatoire pour LightGBM)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement des corpus NLP pour TextBlob
RUN python -m textblob.download_corpora && \
    if [ -d "/root/nltk_data" ]; then mv /root/nltk_data /usr/share/nltk_data; fi

# Copie du code source complet
COPY . .

# Création des dossiers de persistance
RUN mkdir -p /app/models /app/data

# Exposition du port Streamlit par défaut
EXPOSE 8501

# Healthcheck dynamique
HEALTHCHECK CMD sh -c "curl --fail http://localhost:${PORT:-8501}/_stcore/health"

# Commande de lancement dynamique évaluant $PORT si fourni par l'hébergeur
CMD ["sh", "-c", "streamlit run xgboost_trader.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
