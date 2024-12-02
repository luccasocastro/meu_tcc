# Importações
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funções de pré-processamento


def remove_nan_records(df):
    """Remove linhas contendo valores NaN."""
    return df.dropna()


def remove_infinity_records(df):
    """Remove linhas com valores infinitos."""
    df_cleaned = df[~df.isin([np.inf, -np.inf, "Infinity"]).any(axis=1)]
    return df_cleaned


def normalize_data(df):
    """Normaliza colunas numéricas do DataFrame."""
    df['P/L'] = df['P/L'].apply(pd.to_numeric, errors='coerce')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def ajustar_outliers(dados):
    """Ajusta outliers em um array de dados."""
    for _ in range(20):
        mean = np.mean(dados)
        std_dev = np.std(dados)
        dados = np.clip(dados, mean - 3 * std_dev, mean + 3 * std_dev)
    return dados


def remove_duplicate_tickers(df):
    """Remove duplicados com base na coluna 'Ticker'."""
    if 'Ticker' not in df.columns:
        raise ValueError(
            "O DataFrame deve conter uma coluna chamada 'Ticker'.")
    return df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)

# Funções de coleta e processamento de dados


def get_fundamental_data(tickers):
    """Coleta dados fundamentalistas de empresas listadas."""
    data_list = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data = {
                "Ticker": ticker,
                "Empresa": info.get("longName", "N/A"),
                "Setor": info.get("sector", "N/A"),
                "Indústria": info.get("industry", "N/A"),
                "P/L": info.get("trailingPE"),
                "P/VP": info.get("priceToBook"),
                "ROE (%)": info.get("returnOnEquity", None) * 100 if info.get("returnOnEquity") else None,
                "Dívida/Patrimônio": info.get("debtToEquity"),
                "Margem Líquida (%)": info.get("profitMargins", None) * 100 if info.get("profitMargins") else None,
                "Valor de Mercado (Bilhões)": info.get("marketCap", None) / 1e9 if info.get("marketCap") else None,
            }
            data_list.append(data)
        except Exception as e:
            print(f"Erro ao obter dados para {ticker}: {e}")
    df = pd.DataFrame(data_list)
    df = remove_nan_records(df)
    df = remove_infinity_records(df)
    return df


def get_dividend_yield(tickers):
    """Coleta informações de Dividend Yield para cada ticker."""
    results = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker)
            dy = data.info.get('dividendYield', 0.0)
            results.append({"Ticker": ticker, "Dividend_Yield": dy * 100})
        except Exception as e:
            print(f"Erro ao processar {ticker}: {e}")
            results.append({"Ticker": ticker, "Dividend_Yield": None})
    return pd.DataFrame(results)

# Função de redução de dimensionalidade


def reduzir_dimensoes_svd(df, n_dim=3):
    """Reduz a dimensionalidade usando SVD."""
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    reduced_data = svd.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f'Dim_{i+1}' for i in range(n_dim)])

# Função principal de clustering


def realizar_clustering(df):
    """Realiza clustering nos dados fornecidos."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = ajustar_outliers(df[col].values)
    df = normalize_data(df)
    maior_silhouette = -1
    melhor_k = 2
    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df[numeric_columns])
        silhouette_avg = silhouette_score(df[numeric_columns], labels)
        if silhouette_avg > maior_silhouette:
            maior_silhouette = silhouette_avg
            melhor_k = k
    kmeans = KMeans(n_clusters=melhor_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[numeric_columns])
    return df, kmeans

# Gráfico 3D


def plot_3d_clusters(df, kmeans):
    """Plota os clusters em 3D."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = df['P/L'], df['P/VP'], df['ROE (%)']
    ax.scatter(x, y, z, c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               c='red', s=200, marker='X', label='Centróides')
    ax.set_xlabel('P/L')
    ax.set_ylabel('P/VP')
    ax.set_zlabel('ROE (%)')
    ax.set_title('Clustering 3D')
    plt.show()


# Execução principal
if __name__ == "__main__":
    tickers = [
        "PETR4.SA", "VALE3.SA", "ABEV3.SA", "ITUB4.SA", "BBDC4.SA",
        "B3SA3.SA", "WEGE3.SA", "EQTL3.SA", "RADL3.SA", "BBAS3.SA",
        "ITSA4.SA", "JBSS3.SA", "RENT3.SA", "HAPV3.SA", "LREN3.SA",
        "SUZB3.SA", "KLBN11.SA", "GGBR4.SA", "ENBR3.SA", "CSNA3.SA",
        "ELET3.SA", "TAEE11.SA", "CMIG4.SA", "BRKM5.SA", "EMBR3.SA",
        "CPLE6.SA", "AZUL4.SA", "CCRO3.SA", "PRIO3.SA", "BRAP4.SA",
        "GOLL4.SA", "YDUQ3.SA", "HYPE3.SA", "TIMS3.SA", "FLRY3.SA",
        "CRFB3.SA", "MULT3.SA", "MRFG3.SA", "ALPA4.SA", "EGIE3.SA",
        "BPAC11.SA", "BRML3.SA", "TOTS3.SA", "COGN3.SA", "USIM5.SA",
        "CSAN3.SA", "BRFS3.SA", "IGTI11.SA", "SEER3.SA", "CIEL3.SA",
        "AALR3.SA", "AMAR3.SA", "ALSO3.SA", "ARZZ3.SA", "BEEF3.SA",
        "BBSE3.SA", "BIDI11.SA", "BIDI3.SA", "BIDI4.SA", "BMGB4.SA",
        "BOVA11.SA", "BPAN4.SA", "BRDT3.SA", "BRPR3.SA", "BTOW3.SA",
        "CAML3.SA", "CEPE3.SA", "CGAS5.SA", "CIEL3.SA", "CLSC3.SA",
        "CMIN3.SA", "CNTO3.SA", "COGN3.SA", "CRPG5.SA", "CTSA3.SA",
        "CYRE3.SA", "DIRR3.SA", "DOHL3.SA", "ELET6.SA", "ENGI11.SA",
        "ESTC3.SA", "EVEN3.SA", "FESA4.SA", "FLRY3.SA", "FSA3.SA",
        "GENI3.SA", "GNDI3.SA", "GOAU3.SA", "GRND3.SA", "GRAF3.SA",
        "HGTX3.SA", "HYPE3.SA", "IGTA3.SA", "IRBR3.SA", "ITSA4.SA",
        "JHSF3.SA", "KLBN11.SA", "LCAM3.SA", "LIGT3.SA", "LREN3.SA"
    ]
    df = get_fundamental_data(tickers)
    df_dividendos = get_dividend_yield(tickers)
    df = pd.merge(df, df_dividendos, on="Ticker", how="left")
    df = remove_duplicate_tickers(df)
    df, kmeans = realizar_clustering(df)
    plot_3d_clusters(df, kmeans)
    df.to_csv("fundamentalista_resultado.csv", index=False)
