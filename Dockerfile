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

# Téléchargement des corpus NLP pour TextBlob (tolérant aux erreurs de connectivité)
RUN (python -m textblob.download_corpora || true) && \
    if [ -d "/root/nltk_data" ]; then mv /root/nltk_data /usr/share/nltk_data; fi

# Copie du code source complet
COPY . .

# Création des dossiers de persistance
RUN mkdir -p /app/models /app/data /root/.xgboost_trader
VOLUME ["/app/data", "/app/models", "/root/.xgboost_trader"]

# Exposition des ports (8501 standard et 3000 par défaut Dokploy)
EXPOSE 8501 3000

# Commande de lancement universelle avec proxy bi-port
CMD ["python", "dual_port_proxy.py"]
