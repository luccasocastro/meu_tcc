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
    for j in range(20):
        mean = np.mean(dados)
        str_dev = np.std(dados)

        for i in range(len(dados)):
            if dados[i] < mean - 3 * str_dev:
                dados[i] = mean - 3.0 * str_dev
            elif dados[i] > mean + 3 * str_dev:
                dados[i] = mean + 3.0 * str_dev
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
    df = remove_infinity_records(df)
    return df


def get_dividend_yield(tickers):
    """Obtém o Dividend Yield de uma lista de tickers."""
    results = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield')

            if dy is not None:
                dy = dy * 100
            else:
                dy = 0.0

            results.append({"Ticker": ticker, "Dividend_Yield": dy})
        except Exception as e:
            print(f"Erro ao processar o ticker {ticker}: {e}")
            results.append({"Ticker": ticker, "Dividend_Yield": None})
    return pd.DataFrame(results)


def processar_dados(buscar_on_yf=False):
    df = None

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

    if buscar_on_yf:
        df = get_fundamental_data(tickers)
        df.to_csv("codigo/arquivos/fundamentos.csv")
        df = df.drop(columns=['Dívida/Patrimônio',
                              'Margem Líquida (%)', 'Valor de Mercado (Bilhões)'])
    else:
        df = pd.read_csv(
            "codigo/arquivos/fundamentos.csv").drop(columns=['Unnamed: 0'])
        df = df.drop(columns=['Dívida/Patrimônio',
                              'Margem Líquida (%)', 'Valor de Mercado (Bilhões)'])
    return df


# Funções para análise e visualização de dados

def reduzir_dimensoes_svd(df, n_dim=3):
    """Reduz as dimensões do DataFrame usando SVD."""
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    reduced_data = svd.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f'Dim_{i+1}' for i in range(n_dim)])


def unsupervised_rfe(df, n_features_to_select=3, step=1, random_state=42):
    """
    Implementação de RFE adaptado para clustering não supervisionado.

    Args:
        df: DataFrame com as features numéricas
        n_features_to_select: Número de features a serem selecionadas
        step: Número de features a serem removidas em cada iteração
        random_state: Seed para reprodutibilidade

    Returns:
        DataFrame com as features selecionadas
        Lista com a ordem de eliminação das features
    """
    X = df.copy()
    features = list(X.columns)
    features_ranking = []

    # Encontrar o número ótimo de clusters usando silhouette score
    def find_optimal_k(X):
        best_k, best_score = -1, -1
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    while len(features) > n_features_to_select:
        # Encontrar número ótimo de clusters para o conjunto atual de features
        optimal_k = find_optimal_k(X[features])

        # Calcular importância das features
        feature_scores = {}
        for feature in features:
            # Criar conjunto sem a feature
            temp_features = [f for f in features if f != feature]

            # Calcular qualidade do clustering sem a feature
            kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
            labels = kmeans.fit_predict(X[temp_features])
            score = silhouette_score(X[temp_features], labels)

            feature_scores[feature] = score

        # Ordenar features pela importância (menos importante tem maior score quando removida)
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Remover as features menos importantes
        features_to_remove = [f[0] for f in sorted_features[:step]]
        for f in features_to_remove:
            features.remove(f)
            features_ranking.append(f)

        print(f"Iteração: Mantidas {len(features)} features")
        print(f"Removidas: {features_to_remove}")

    print("\nFeatures selecionadas:", features)
    print("Ordem de eliminação:", features_ranking)

    return X[features], features_ranking


if __name__ == "__main__":
    df = processar_dados(buscar_on_yf=False)
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Ajuste de outliers e normalização
    for col in numeric_columns:
        df[col] = ajustar_outliers(df[col].values)
    df = normalize_data(df)

    # Teste seleção de features
    df_features = df[numeric_columns]
    selected_features, features_ranking = unsupervised_rfe(
        df_features, n_features_to_select=3)

    # Agora usamos apenas as features selecionadas
    print("\nAplicando clustering com as features selecionadas:")
    print(selected_features.columns)

    # Redução de dimensões e clustering
    # df_reduzido = df[numeric_columns]
    melhor_silhouette, melhor_k = -1.0, -1
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(selected_features)
        silhouette_avg = silhouette_score(selected_features, labels)

        if silhouette_avg > melhor_silhouette:
            melhor_silhouette, melhor_k = silhouette_avg, k

    # Aplicar KMeans com o número ótimo de clusters
    kmeans = KMeans(n_clusters=melhor_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(selected_features)

    # Imprimindo os clusters atribuídos
    print("Clusters atribuídos aos dados:")
    print(df[['Ticker', 'Cluster']])

    # Visualização em 3D
    if len(selected_features.columns) >= 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = selected_features.iloc[:,
                                         0], selected_features.iloc[:, 1], selected_features.iloc[:, 2]

        ax.scatter(x, y, z, c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
        ax.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   kmeans.cluster_centers_[:, 2],
                   c='red', s=200, marker='X', label='Centróides')

        ax.set_xlabel(selected_features.columns[0])
        ax.set_ylabel(selected_features.columns[1])
        ax.set_zlabel(selected_features.columns[2])
        ax.set_title('Clustering com Features Selecionadas por RFE')

        # Criando legenda dos clusters
        from matplotlib.lines import Line2D
        import matplotlib.cm as cm

        n_clusters = len(np.unique(df['Cluster']))
        cmap = plt.cm.get_cmap('viridis', n_clusters)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                   markerfacecolor=cmap(i), markersize=10)
            for i in range(n_clusters)
        ]
        legend_elements.append(
            Line2D([0], [0], marker='X', color='w', label='Centróides',
                   markerfacecolor='red', markersize=12)
        )
        ax.legend(handles=legend_elements)

        plt.show()
    else:
        print("Visualização 3D não disponível - menos de 3 features selecionadas")

    print("\nResultados finais:")
    print(df[['Ticker', 'Cluster'] + list(selected_features.columns)])

    # Integração com dados de Dividend Yield
    # df_dividendos = get_dividend_yield(tickers)
    # df = pd.merge(df, df_dividendos, on="Ticker", how="left")
    # df = remove_duplicate_tickers(df)
    # print(df)
