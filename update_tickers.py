import os
import pandas as pd
import json
import requests
from io import StringIO

# Liste d'ETFs majeurs US pour offrir le choix maximum à l'utilisateur
MAJOR_ETFS = {
    "SPDR S&P 500 ETF Trust (SPY)": "NASDAQ:SPY",
    "Invesco QQQ Trust (QQQ)": "NASDAQ:QQQ",
    "SPDR Dow Jones Industrial Average ETF Trust (DIA)": "NASDAQ:DIA",
    "iShares Russell 2000 ETF (IWM)": "NASDAQ:IWM",
    "SPDR Gold Shares (GLD)": "NASDAQ:GLD",
    "iShares Silver Trust (SLV)": "NASDAQ:SLV",
    "iShares 20+ Year Treasury Bond ETF (TLT)": "NASDAQ:TLT",
    "iShares 7-10 Year Treasury Bond ETF (IEF)": "NASDAQ:IEF",
    "Financial Select Sector SPDR Fund (XLF)": "NASDAQ:XLF",
    "Technology Select Sector SPDR Fund (XLK)": "NASDAQ:XLK",
    "Energy Select Sector SPDR Fund (XLE)": "NASDAQ:XLE",
    "Health Care Select Sector SPDR Fund (XLV)": "NASDAQ:XLV",
    "Industrial Select Sector SPDR Fund (XLI)": "NASDAQ:XLI",
    "Consumer Discretionary Select Sector SPDR Fund (XLY)": "NASDAQ:XLY",
    "Consumer Staples Select Sector SPDR Fund (XLP)": "NASDAQ:XLP",
    "Materials Select Sector SPDR Fund (XLB)": "NASDAQ:XLB",
    "Real Estate Select Sector SPDR Fund (XLRE)": "NASDAQ:XLRE",
    "Utilities Select Sector SPDR Fund (XLU)": "NASDAQ:XLU",
    "Ark Innovation ETF (ARKK)": "NASDAQ:ARKK",
    "Vanguard Total Stock Market ETF (VTI)": "NASDAQ:VTI",
    "Vanguard FTSE Developed Markets ETF (VEA)": "NASDAQ:VEA",
    "Vanguard Real Estate ETF (VNQ)": "NASDAQ:VNQ",
    "iShares iBoxx $ Investment Grade Corporate Bond ETF (LQD)": "NASDAQ:LQD",
    "iShares iBoxx $ High Yield Corporate Bond ETF (HYG)": "NASDAQ:HYG",
    "ProShares Short S&P500 (SH)": "NASDAQ:SH",
    "ProShares UltraPro QQQ (TQQQ)": "NASDAQ:TQQQ",
    "ProShares UltraPro Short QQQ (SQQQ)": "NASDAQ:SQQQ"
}

def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text

def get_table_with_columns(dfs, req_cols):
    for df in dfs:
        cols = set(df.columns)
        if all(c in cols for c in req_cols):
            return df
    raise ValueError(f"Table contenant les colonnes {req_cols} non trouvée.")

def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Symbol', 'Security'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Symbol']).strip().replace('.', '-')
        name = str(row['Security']).strip()
        tickers[f"{name} (US)"] = f"NASDAQ:{symbol}"
    return tickers

def fetch_cac40():
    url = "https://en.wikipedia.org/wiki/CAC_40"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Ticker', 'Company'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Ticker']).strip().replace('.PA', '')
        name = str(row['Company']).strip()
        tickers[f"{name} (FR)"] = f"EPA:{symbol}"
    return tickers

def fetch_dax():
    url = "https://en.wikipedia.org/wiki/DAX"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Ticker', 'Company'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Ticker']).strip().replace('.DE', '')
        name = str(row['Company']).strip()
        tickers[f"{name} (DE)"] = f"FRA:{symbol}"
    return tickers

def main():
    try:
        print("Téléchargement du S&P 500...")
        sp500 = fetch_sp500()
        print("Téléchargement du CAC 40...")
        cac40 = fetch_cac40()
        print("Téléchargement du DAX...")
        dax = fetch_dax()

        all_tickers = {"--- Saisir manuellement ---": "CUSTOM"}
        all_tickers.update(MAJOR_ETFS)
        all_tickers.update(sp500)
        all_tickers.update(cac40)
        all_tickers.update(dax)

        code = []
        code.append("# Base de données locale des grandes actions (EU & US)")
        code.append("# Format attendu par notre convertisseur : Google Finance (Bourse:Symbole)")
        code.append("")
        code.append("import os")
        code.append("import csv")
        code.append("import difflib")
        code.append("")
        code.append("def _get_best_match(name):")
        code.append("    for k in MAJOR_STOCKS.keys():")
        code.append("        if k == '--- Saisir manuellement ---': continue")
        code.append("        clean_k = k.split(' (')[0].lower()")
        code.append("        if clean_k in name.lower() or name.lower() in clean_k:")
        code.append("            return k")
        code.append("    best_match = None")
        code.append("    best_score = 0")
        code.append("    for k in MAJOR_STOCKS.keys():")
        code.append("        if k == '--- Saisir manuellement ---': continue")
        code.append("        clean_k = k.split(' (')[0]")
        code.append("        score = difflib.SequenceMatcher(None, name.lower(), clean_k.lower()).ratio()")
        code.append("        if score > best_score:")
        code.append("            best_score = score")
        code.append("            best_match = k")
        code.append("    if best_score > 0.65:")
        code.append("        return best_match")
        code.append("    return None")
        code.append("")
        code.append("def _load_dynamic_watchlists():")
        code.append("    base_dir = os.path.dirname(os.path.abspath(__file__))")
        code.append("    watchlists_dir = os.path.join(base_dir, 'WatchList')")
        code.append("    portfolios = {}")
        code.append("    if os.path.exists(watchlists_dir):")
        code.append("        import pandas as pd")
        code.append("        for file in os.listdir(watchlists_dir):")
        code.append("            if not file.endswith('.csv'): continue")
        code.append("            list_name = f'📂 {file.split(\"_Watchlist\")[0]}'")
        code.append("            stocks = []")
        code.append("            try:")
        code.append("                filepath = os.path.join(watchlists_dir, file)")
        code.append("                df = pd.read_csv(filepath, sep=';', encoding='utf-8', on_bad_lines='skip')")
        code.append("                if len(df.columns) < 2:")
        code.append("                    df = pd.read_csv(filepath, sep=',', encoding='utf-8', on_bad_lines='skip')")
        code.append("                for _, row in df.iterrows():")
        code.append("                    name = str(row.get('Name', row.get('Nom', 'Unknown'))).strip()")
        code.append("                    symbol = str(row.get('Symbol', row.get('Symbole', ''))).strip()")
        code.append("                    if not symbol or symbol.lower() in ('nan', 'none'):")
        code.append("                        continue")
        code.append("                    if symbol.endswith(('.O', '.K', '.N', '.OQ')):")
        code.append("                        clean_symbol = symbol.rsplit('.', 1)[0]")
        code.append("                    else:")
        code.append("                        clean_symbol = symbol")
        code.append("                    display_name = f'{name} ({clean_symbol})'")
        code.append("                    MAJOR_STOCKS[display_name] = clean_symbol")
        code.append("                    if display_name not in stocks:")
        code.append("                        stocks.append(display_name)")
        code.append("                if stocks:")
        code.append("                    portfolios[list_name] = stocks")
        code.append("            except Exception:")
        code.append("                pass")
        code.append("    return portfolios")
        code.append("")
        code.append("MAJOR_STOCKS = {")
        for i, (name, symbol) in enumerate(all_tickers.items()):
            safe_name = name.replace('"', '\\"')
            comma = "," if i < len(all_tickers) - 1 else ""
            code.append(f'    "{safe_name}": "{symbol}"{comma}')
        code.append("}")
        code.append("")
        code.append("PREDEFINED_PORTFOLIOS = {")
        code.append("    'Sélection Manuelle': [],")
        code.append("    'Portefeuille Warren Buffett': [")
        code.append("        'Apple Inc. (US)', 'Bank of America (US)', 'American Express (US)', ")
        code.append("        'Coca-Cola Company (The) (US)', 'Chevron Corporation (US)', ")
        code.append("        'Occidental Petroleum (US)', 'Kraft Heinz (US)'")
        code.append("    ],")
        code.append("    'Géants de la Tech (Magnificent 7)': [")
        code.append("        'Apple Inc. (US)', 'Microsoft (US)', 'Alphabet Inc. (Class A) (US)', ")
        code.append("        'Amazon (US)', 'Nvidia (US)', 'Meta Platforms (US)', 'Tesla, Inc. (US)'")
        code.append("    ],")
        code.append("    'Tendances Positives US (S&P 500 Growth)': [")
        code.append("        'Nvidia (US)', 'Broadcom (US)', 'Lilly (Eli) (US)', ")
        code.append("        'Advanced Micro Devices (US)', 'CrowdStrike (US)', 'Palantir Technologies (US)'")
        code.append("    ],")
        code.append("    'Top Europe (Luxe & Tech)': [")
        code.append("        'LVMH (FR)', 'L\\'Oréal (FR)', 'SAP (DE)', ")
        code.append("        'Schneider Electric (FR)', 'Siemens (DE)', 'Hermès (FR)'")
        code.append("    ],")
        code.append("    'Tendances Asie / Chine (Tech & EVs)': [")
        code.append("        'BABA', 'TCEHY', 'BYDDY', 'BIDU', 'JD'")
        code.append("    ],")
        code.append("    'ETFs Majeurs US (Indices & Secteurs)': [")
        code.append("        'SPDR S&P 500 ETF Trust (SPY)', 'Invesco QQQ Trust (QQQ)', 'iShares Russell 2000 ETF (IWM)', ")
        code.append("        'Technology Select Sector SPDR Fund (XLK)', 'Financial Select Sector SPDR Fund (XLF)', ")
        code.append("        'Energy Select Sector SPDR Fund (XLE)', 'Health Care Select Sector SPDR Fund (XLV)', ")
        code.append("        'Ark Innovation ETF (ARKK)'")
        code.append("    ]")
        code.append("}")
        code.append("")
        code.append("# Chargement dynamique des portefeuilles depuis le dossier WatchList")
        code.append("try:")
        code.append("    DYNAMIC_PORTFOLIOS = _load_dynamic_watchlists()")
        code.append("    PREDEFINED_PORTFOLIOS.update(DYNAMIC_PORTFOLIOS)")
        code.append("except Exception as e:")
        code.append("    pass")
        code.append("")

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers_db.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(code))
        print(f"tickers_db.py généré avec succès à l'adresse : {out_path}")
    except Exception as e:
        print(f"Erreur lors de la mise à jour : {e}")

if __name__ == '__main__':
    main()
