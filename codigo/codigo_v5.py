# Importação de bibliotecas
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import davies_bouldin_score
import time

# Funções para limpeza de dados
# Lista de tickers do iBovespa 1T/2025
TICKERS = [
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
    "JHSF3.SA", "KLBN11.SA", "LCAM3.SA", "LIGT3.SA", "LREN3.SA",
    "SMFT3.SA"
]


def remove_nan_records(df):
    """Remove linhas que contêm valores NaN."""
    return df.dropna()


def remove_infinity_records(df):
    df_replaced = df.replace("Infinity", np.inf)
    mask_inf = df_replaced.isin([np.inf, -np.inf]).any(axis=1)
    df_cleaned = df_replaced.loc[~mask_inf]
    return df_cleaned


def normalize_data(df):
    """Normaliza apenas as colunas numéricas do DataFrame usando Min-Max Scaling,
    preservando as demais colunas sem modificá-las."""

    df_copy = df.copy()

    # Identifica colunas numéricas
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns

    # Aplica Min-Max Scaling apenas nas colunas numéricas
    scaler = MinMaxScaler()
    df_copy[numeric_columns] = scaler.fit_transform(df_copy[numeric_columns])

    return df_copy


def ajustar_outliers(dados):
    dados = np.array(dados, dtype=np.float64)

    for _ in range(10):
        mean = np.nanmean(dados)
        std_dev = np.nanstd(dados)

        # Se std_dev for zero ou NaN, interrompe o ajuste
        if np.isnan(std_dev) or std_dev == 0:
            break

        lower = mean - 3 * std_dev
        upper = mean + 3 * std_dev

        dados = np.where(dados < lower, lower, dados)
        dados = np.where(dados > upper, upper, dados)

    return dados


def remove_duplicate_tickers(df):
    """Remove duplicatas com base na coluna 'Ticker'."""
    if 'ticker' not in df.columns:
        raise ValueError(
            "O DataFrame deve conter uma coluna chamada 'Ticker'.")
    return df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)

# Funções para obtenção e manipulação de dados fundamentalistas


def get_fundamental_data1():
    # Campos de interesse
    campos_fundamentalistas = [
        'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 'pegRatio',
        'priceToBook', 'bookValue', 'enterpriseToRevenue', 'enterpriseToEbitda',
        'profitMargins', 'grossMargins', 'operatingMargins', 'returnOnAssets',
        'returnOnEquity', 'revenuePerShare', 'earningsPerShare', 'trailingEps',
        'forwardEps', 'totalRevenue', 'ebitda', 'debtToEquity', 'currentRatio',
        'quickRatio', 'totalCash', 'totalDebt', 'freeCashflow'
    ]

    # Lista para armazenar os dados
    dados_fundamentalistas = []

    # Coleta dos dados
    for ticker in TICKERS:
        print(f"Coletando dados de {ticker}...")
        try:
            ativo = yf.Ticker(ticker)
            info = ativo.info
            dados = {'ticker': ticker}
            for campo in campos_fundamentalistas:
                dados[campo] = info.get(campo, None)
            dados_fundamentalistas.append(dados)
        except Exception as e:
            print(f"Erro ao coletar {ticker}: {e}")
        time.sleep(1)  # Pausa para evitar bloqueio da API

    # Criação do DataFrame
    df = pd.DataFrame(dados_fundamentalistas)

    # Salva para CSV
    # df.to_csv("dados_fundamentalistas.csv", index=False)

    return df


def get_dividend_yield():
    """Obtém o Dividend Yield de uma lista de tickers."""
    results = []
    for ticker in TICKERS:
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


def calcular_valorizacao(acoes, data_inicio, data_fim):
    dados = yf.download(acoes, start=data_inicio, end=data_fim)['Close']

    valorizacao = {}
    for acao in acoes:
        try:
            preco_inicial = dados[acao].iloc[0]
            preco_final = dados[acao].iloc[-1]
            variacao_percentual = (
                (preco_final - preco_inicial) / preco_inicial) * 100
            valorizacao[acao] = variacao_percentual
        except Exception as e:
            print(f"Erro ao calcular valorização de {acao}: {e}")
            valorizacao[acao] = None

    df = pd.DataFrame.from_dict(
        valorizacao, orient='index', columns=['Valorização (%)'])
    df.index.name = 'ticker'
    df.reset_index(inplace=True)
    return df.round(2)


def get_fundamental_data():
    """Obtém dados fundamentalistas de uma lista de tickers."""
    data_list = []
    for ticker in TICKERS:
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


def obter_dados_completos(df, data_inicio, data_fim):
    df_valor = calcular_valorizacao(TICKERS, data_inicio, data_fim)

    df_completo = pd.merge(df, df_valor, on='ticker', how='left')

    # Salvar opcional
    df_completo.to_csv(
        "dados_completos_e_limpos_com_valorizacao.csv", index=False)


def remove_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.iloc[:, 1:]

    # Cria uma máscara para as linhas onde todas as colunas da seleção são NaN
    mask = ~subset.isna().all(axis=1)

    # Aplica a máscara ao DataFrame original
    df = df[mask].reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    return df


def processar_dados(buscar_on_yf=False):
    df = None

    if buscar_on_yf:
        df = get_fundamental_data1()
    else:
        df = pd.read_csv("dados_completos.csv")

    df = remove_empty_rows_and_columns(df)
    df.to_csv("dados_completos_e_limpos.csv")
    return df


# Funções para análise e visualização de dados

def reduzir_dimensoes_svd(df, n_dim=3):
    """Reduz as dimensões do DataFrame usando SVD."""
    # svd = TruncatedSVD(n_components=n_dim, random_state=42)
    # reduced_data = svd.fit_transform(df)
    # return pd.DataFrame(reduced_data, columns=[f'Dim_{i+1}' for i in range(n_dim)])

    svd = TruncatedSVD(n_components=n_dim)
    dados_reduzidos = svd.fit_transform(df)
    return dados_reduzidos


def unsupervised_rfe(df, n_features_to_select=3, step=1, random_state=42):
    """
    Implementação de RFE adaptado para clustering não supervisionado.
    """
    X = df.copy()
    features = list(X.columns)
    features_ranking = []

    def find_optimal_k(X_subset):
        best_k, best_score = -1, -1
        for k in range(2, min(10, len(X_subset))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                labels = kmeans.fit_predict(X_subset)
                score = silhouette_score(X_subset, labels)
                if np.isnan(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k if best_k > 1 else 2

    while len(features) > n_features_to_select:
        optimal_k = find_optimal_k(X[features])

        feature_scores = {}
        for feature in features:
            temp_features = [f for f in features if f != feature]
            try:
                kmeans = KMeans(n_clusters=optimal_k,
                                random_state=random_state)
                labels = kmeans.fit_predict(X[temp_features])
                score = silhouette_score(X[temp_features], labels)
                if np.isnan(score):
                    score = -1  # Penaliza features que resultam em clustering inválido
                feature_scores[feature] = score
            except Exception:
                feature_scores[feature] = -1

        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True)
        features_to_remove = [f[0] for f in sorted_features[:min(
            step, len(features) - n_features_to_select)]]

        for f in features_to_remove:
            features.remove(f)
            features_ranking.append(f)

        print(f"Iteração: Mantidas {len(features)} features")
        print(f"Removidas: {features_to_remove}")

    print("\nFeatures selecionadas:", features)
    print("Ordem de eliminação:", features_ranking)

    return df[features], features_ranking


def buscar_dividend_yield_por_ticker(df):
    """
    Busca o Dividend Yield para cada ticker presente na coluna 'ticker' do DataFrame.
    Retorna um DataFrame com os tickers e seus respectivos dividend yields.
    """
    dividend_yields = []

    for ticker in df['ticker'].unique():
        try:
            ticker_formatado = ticker if ticker.endswith(
                '.SA') else f"{ticker}.SA"
            stock = yf.Ticker(ticker_formatado)
            info = stock.info

            dy = info.get('dividendYield', None)
            # if dy is not None:
            # dy = dy * 100  # Converte para percentual
            dividend_yields.append({'ticker': ticker, 'dividend_yield': dy})
            print(f'Dividend yield de {ticker} obtido!')
        except Exception as e:
            print(f"Erro ao buscar dados de {ticker}: {e}")
            dividend_yields.append({'ticker': ticker, 'dividend_yield': None})

    return pd.DataFrame(dividend_yields)


def execution(df, numeric_columns, step):
    index_silhouette = []
    index_bouldin = []

    for i in range(1, 11):
        if df.empty:
            print("\nDataFrame está vazio após limpeza.")
        else:
            # Ajuste de outliers
            print("\nAjustando outliers...")
            for col in numeric_columns:
                df[col] = ajustar_outliers(df[col].values)

            print("Normalização com min-max...")
            df = normalize_data(df)

            print("\nExecutando teste de seleção de features com RFE...")
            # Teste seleção de features
            selected_features, features_ranking = unsupervised_rfe(
                df[numeric_columns], n_features_to_select=i)

            # Agora usamos apenas as features selecionadas
            print("\nAplicando clustering com as features selecionadas:")
            print(selected_features.columns)

            # Redução de dimensões e clustering
            melhor_silhouette, melhor_k = -1.0, -1
            melhor_davies_bouldin = 0.0

            for k in range(2, 10):
                kmeans = KMeans(n_clusters=k)  # , random_state=42)
                labels = kmeans.fit_predict(selected_features)
                silhouette_avg = silhouette_score(selected_features, labels)

                if silhouette_avg > melhor_silhouette:
                    melhor_silhouette, melhor_k = silhouette_avg, k
                    db_index = davies_bouldin_score(selected_features, labels)
                    melhor_davies_bouldin = db_index

            # Aplicar KMeans com o número ótimo de clusters
            kmeans = KMeans(n_clusters=melhor_k)  # , random_state=42)
            df['Cluster'] = kmeans.fit_predict(selected_features)
            print(f"Melhor Davis Bouldin: {melhor_davies_bouldin}")
            print(f"Melhor Silhouette: {melhor_silhouette}")
            index_silhouette.append(melhor_silhouette)
            index_bouldin.append(melhor_davies_bouldin)

            # df.to_csv(f"result_step_{step}_iteration_{i}.csv")

        media_silhouette = 0
        media_bouldin = 0
        for index, sil in enumerate(index_silhouette):
            media_silhouette += sil

        for index, bou in enumerate(index_bouldin):
            media_bouldin += bou

    return (media_silhouette/10), (media_bouldin/10)


# Exemplo de uso no fluxo principal
if __name__ == "__main__":
    index_silhouette = []
    index_bouldin = []
    features_selected = []

    df = processar_dados(buscar_on_yf=True)

    # print("\nSubstituindo NANs por medianas...", df.shape)
    # Em vez de dropna, vamos substituir NaNs pelas medianas para evitar perda excessiva
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # print("\nRemovendo linhas com infinitos...", df.shape)
    df = remove_infinity_records(df)

    for i in range(1, 11):
        silhouette, bouldin = execution(
            df=df, numeric_columns=numeric_columns, step=i)
        index_silhouette.append(silhouette)
        index_bouldin.append(bouldin)

    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("\nResultados finais:")
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=")

    obter_dados_completos(df, '2019-06-06', '2024-06-06')

    media_silhouette = 0
    media_bouldin = 0

    for index, i in enumerate(index_silhouette):
        print(f"\nSILHOUETTE ITERAÇÃO {index+1}: {i:.5f}")
        media_silhouette += i

    for index, j in enumerate(index_bouldin):
        print(f"\nBOULDIN ITERAÇÃO {index+1}: {j:.5f}")
        media_bouldin += j

    print(
        f"\nMÉDIA SILHOUETTE: {media_silhouette/10:.5f}, MÉDIA BOULDIN: {media_bouldin/10:.5f}")

    for index, feat in enumerate(features_selected):
        print(f"Iteração {index+1}: {feat.columns}")
