# Importações
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funções de Pré-processamento

def remove_nan_records(df):
    """Remove linhas contendo valores NaN."""
    return df.dropna()

def remove_infinity_records(df):
    """Remove linhas com valores infinitos."""
    df_cleaned = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    df_cleaned = df[~df.isin(["Infinity"]).any(axis=1)]
    return df_cleaned

def normalize_data(df):
    """Normaliza colunas numéricas do DataFrame."""
    
    # Converter colunas numéricas para o tipo float, ignorando erros
    df['P/L'] = df['P/L'].apply(pd.to_numeric, errors='coerce')
    # Selecionar apenas colunas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    # Aplicar Normalização Z-Score
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def ajustar_outliers(dados):
    """Ajustar possíveis outliers"""
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
    """Remove duplicados com base na coluna 'Ticker'."""
    if 'Ticker' not in df.columns:
        raise ValueError(
            "O DataFrame deve conter uma coluna chamada 'Ticker'.")
    return df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)

# Funções de coleta e processamento de dados

def get_fundamental_data(tickers):
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
    print("Buscando dividendos...")
    results = []
    for ticker in tickers:
        try:
            # Obter dados do ticker
            data = yf.Ticker(ticker)
            info = data.info
            
            # Coletar o Dividend Yield, se disponível
            dy = info.get('dividendYield')
            if dy is not None:
                dy = dy * 100  # Converter para percentual
            else:
                dy = 0.0  # Caso não tenha DY, considerar 0
            
            results.append({"Ticker": ticker, "Dividend_Yield": dy})
        except Exception as e:
            print(f"Erro ao processar o ticker {ticker}: {e}")
            results.append({"Ticker": ticker, "Dividend_Yield": None})
    
    # Converter os resultados em DataFrame
    df = pd.DataFrame(results)
    return df

# Função de redução de dimensionalidade

def reduzir_dimensoes_svd(df, n_dim=3):
    """Reduz a dimensionalidade usando SVD."""
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    reduced_data = svd.fit_transform(df)
    return pd.DataFrame(reduced_data, columns=[f'Dim_{i+1}' for i in range(n_dim)])

# Exemplo de uso
# Lista de ativos do iBovespa
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
# # df.to_csv("fundamentalista.csv")
# print(df)

# Busca os dados direto do csv
df = pd.read_csv("fundamentos.csv")
df = df.drop(columns=['Unnamed: 0'])
# df = df.drop(columns=['P/VP'])
df = df.drop(columns=['Dívida/Patrimônio'])
df = df.drop(columns=['Margem Líquida (%)'])
df = df.drop(columns=['Valor de Mercado (Bilhões)'])
print(df)

# Aplicando a função ajustar_outliers a cada coluna numérica separadamente
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Iterando sobre as colunas numéricas e ajustando os outliers
for col in numeric_columns:
    df[col] = ajustar_outliers(df[col].values)

# Normalizando colunas numéricas
df = normalize_data(df)

# Selecionando as colunas numéricas para a análise de clustering
# Aplicando SVD ao DataFrame
# df_reduzido = reduzir_dimensoes_svd(df[numeric_columns])
df_reduzido = df[numeric_columns]

# Encontrando o número ideal de clusters utilizando o coeficiente de silhueta
maior = -1.0
indice = -1
for i in range(3, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_reduzido)

    # Avaliar com o coeficiente de silhueta
    silhouette_avg = silhouette_score(df_reduzido, kmeans.fit_predict(df_reduzido))

    if silhouette_avg > maior:
        maior = silhouette_avg
        indice = i

# Aplicando o algoritmo K-Means com o número ideal de clusters
kmeans = KMeans(n_clusters=indice)
kmeans.fit(df_reduzido)

# Adicionando os clusters ao DataFrame original
df['Cluster'] = kmeans.labels_

# Imprimindo os clusters atribuídos
print("Clusters atribuídos aos dados:")
print(df[['Ticker', 'Cluster']])

# Visualizando os dados reduzidos em 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Usando as dimensões reduzidas para o gráfico
# x = df_reduzido['Dim_1']
# y = df_reduzido['Dim_2']
# z = df_reduzido['Dim_3']
x = df_reduzido['P/L']
y = df_reduzido['P/VP']
z = df_reduzido['ROE (%)']

# Plotando os pontos
# c = np.random.rand(len(x))  # Um valor de cor para cada ponto
ax.scatter(x, y, z, c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)

# Obtendo os centróides dos clusters
centroides = kmeans.cluster_centers_

# Plotando os centróides
ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], 
           c='red', s=200, marker='X', label='Centróides')

# Rótulos e título
# ax.set_xlabel('Dim_1')
# ax.set_ylabel('Dim_2')
# ax.set_zlabel('Dim_3')
ax.set_xlabel('P/L')
ax.set_ylabel('P/VP')
ax.set_zlabel('ROE (%)')

ax.set_title('Clustering com Dimensões Reduzidas')

plt.show()

df_dividendos = get_dividend_yield(tickers)
df = pd.merge(df, df_dividendos, on="Ticker", how="left")
df = remove_duplicate_tickers(df)
print(df)

# Salvar CSV com resultado final do processamento
df.to_csv("fundamentalista.csv")
