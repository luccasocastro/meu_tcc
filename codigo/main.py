import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

tickers = [
    "ALOS3.SA", "ABEV3.SA", "ASAI3.SA", "AZUL4.SA", "AZZA3.SA", "B3SA3.SA", "BBSE3.SA", 
    "BBDC4.SA", "BBAS3.SA", "BRAV3.SA", "BRFS3.SA", "BPAC11.SA", "CMIG4.SA", "CPLE6.SA", 
    "CSAN3.SA", "CYRE3.SA", "ELET3.SA", "EMBR3.SA", "ENGI11.SA", "EQTL3.SA", "GGBR4.SA", 
    "NTCO3.SA", "HAPV3.SA", "HYPE3.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", 
    "RENT3.SA", "LREN3.SA", "MGLU3.SA", "MRVE3.SA", "MULT3.SA", "PETR3.SA", "PETR4.SA", 
    "PRIO3.SA", "RADL3.SA", "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "CSNA3.SA", "SUZB3.SA", 
    "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA", "VBBR3.SA", 
    "WEGE3.SA"
]

data = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info
    data[ticker] = {
        'marketCap': info.get('marketCap', 0),
        'priceToBook': info.get('priceToBook', 0),
        'dividendYield': info.get('dividendYield', 0) * 100, 
        'trailingPE': info.get('trailingPE', 0),
        'debtToEquity': info.get('debtToEquity', 0)
    }

df = pd.DataFrame(data).T

df.replace(0, pd.NA, inplace=True)
df.dropna(inplace=True)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

sns.set_theme(style='whitegrid')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='priceToBook', y='dividendYield', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Clusterização de Empresas por Indicadores Fundamentalistas')
plt.xlabel('Price to Book Ratio')
plt.ylabel('Dividend Yield (%)')
# plt.show()
plt.savefig("clusterizacao_empresas.png")

for cluster_id in df['Cluster'].unique():
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['Cluster'] == cluster_id]
    formatted_data = cluster_data.apply(
        lambda x: pd.Series({
            'marketCap': f"R$ {x['marketCap']:,.2f}",
            'priceToBook': f"{x['priceToBook']:.2f}",
            'dividendYield': f"{x['dividendYield']:.2f}%",
            'trailingPE': f"{x['trailingPE']:.2f}",
            'debtToEquity': f"{x['debtToEquity']:.2f}%"
        }), axis=1
    )
    print(formatted_data)

