# Importação de bibliotecas
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Funções para limpeza de dados


def remove_nan_records(df):
    """Remove linhas que contêm valores NaN."""
    return df.dropna()


def remove_infinity_records(df):
    """Remove valores infinitos do DataFrame."""
    df_cleaned = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df_cleaned[~df.isin(["Infinity"]).any(axis=1)]


def normalize_data(df):
    """Normaliza os dados numéricos usando Z-Score."""
    df['P/L'] = df['P/L'].apply(pd.to_numeric, errors='coerce')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def ajustar_outliers(dados):
    """Ajusta outliers baseando-se em média e desvio padrão."""
    for _ in range(20):
        mean, std_dev = np.mean(dados), np.std(dados)
        dados = np.clip(dados, mean - 3 * std_dev, mean + 3 * std_dev)
    return dados


def remove_duplicate_tickers(df):
    """Remove duplicatas com base na coluna 'Ticker'."""
    if 'Ticker' not in df.columns:
        raise ValueError(
            "O DataFrame deve conter uma coluna chamada 'Ticker'.")
    return df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)

# Funções para obtenção e manipulação de dados fundamentalistas


def get_fundamental_data(tickers):
    """Obtém dados fundamentalistas de uma lista de tickers."""
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
                "P/L": info.get("trailingPE", None),
                "P/VP": info.get("priceToBook", None),
                "ROE (%)": info.get("returnOnEquity", None) * 100 if info.get("returnOnEquity") else None,
                "Dívida/Patrimônio": info.get("debtToEquity", None),
                "Margem Líquida (%)": info.get("profitMargins", None) * 100 if info.get("profitMargins") else None,
                "Valor de Mercado (Bilhões)": info.get("marketCap", None) / 1e9 if info.get("marketCap") else None
            }
            data_list.append(data)
        except Exception as e:
            print(f"Erro ao obter dados para o ticker {ticker}: {e}")
    df = pd.DataFrame(data_list)
    df = remove_nan_records(df)
    return remove_infinity_records(df)


def get_dividend_yield(tickers):
    """Obtém o Dividend Yield de uma lista de tickers."""
    results = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield', 0.0) * \
                100 if info.get('dividendYield') else 0.0
            results.append({"Ticker": ticker, "Dividend_Yield": dy})
        except Exception as e:
            print(f"Erro ao processar o ticker {ticker}: {e}")
            results.append({"Ticker": ticker, "Dividend_Yield": None})
    return pd.DataFrame(results)

# Funções para análise e visualização de dados


def reduzir_dimensoes_svd(df, n_dim=3):
    """Reduz as dimensões do DataFrame usando SVD."""
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    reduced_data = svd.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f'Dim_{i+1}' for i in range(n_dim)])


# Processo principal
if __name__ == "__main__":
    # Lista de tickers do iBovespa
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

    # Caso ainda não tenha realizado a busca dos dados e salvo no csv
    # df = get_fundamental_data(tickers)
    # df.to_csv("fundamentos.csv")
    # df = df.drop(columns=['P/VP'])
    # df = df.drop(columns=['Dívida/Patrimônio'])
    # df = df.drop(columns=['Margem Líquida (%)'])
    # df = df.drop(columns=['Valor de Mercado (Bilhões)'])
    # df.to_csv("fundamentalista.csv")
    # print(df)

    # Obtenção e limpeza dos dados
    df = pd.read_csv("codigo/fundamentos.csv").drop(columns=['Unnamed: 0'])
    df = df.drop(columns=['Dívida/Patrimônio',
                 'Margem Líquida (%)', 'Valor de Mercado (Bilhões)'])
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = ajustar_outliers(df[col].values)
    df = normalize_data(df)

    # Redução de dimensões e clustering
    df_reduzido = df[numeric_columns]
    melhor_silhouette, melhor_k = -1.0, -1
    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k)
        silhouette_avg = silhouette_score(
            df_reduzido, kmeans.fit_predict(df_reduzido))
        if silhouette_avg > melhor_silhouette:
            melhor_silhouette, melhor_k = silhouette_avg, k
    kmeans = KMeans(n_clusters=melhor_k)
    df['Cluster'] = kmeans.fit_predict(df_reduzido)

    # Visualização em 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = df_reduzido['P/L'], df_reduzido['P/VP'], df_reduzido['ROE (%)']
    ax.scatter(x, y, z, c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               kmeans.cluster_centers_[:, 2], c='red', s=200, marker='X', label='Centróides')
    ax.set_xlabel('P/L')
    ax.set_ylabel('P/VP')
    ax.set_zlabel('ROE (%)')
    ax.set_title('Clustering com Dimensões Reduzidas')
    plt.show()

    # Integração com dados de Dividend Yield
    # df_dividendos = get_dividend_yield(tickers)
    # df = pd.merge(df, df_dividendos, on="Ticker", how="left")
    # df = remove_duplicate_tickers(df)
    # print(df)
