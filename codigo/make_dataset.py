import yfinance as yf
import pandas as pd

tickers_ibrx50 = [
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
for ticker in tickers_ibrx50:
    stock = yf.Ticker(ticker)
    info = stock.info
    data[ticker] = {
        'marketCap': info.get('marketCap', None),
        'priceToBook': info.get('priceToBook', None),
        'dividendYield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
        'trailingPE': info.get('trailingPE', None),
        'debtToEquity': info.get('debtToEquity', None)
    }

df = pd.DataFrame(data).T

df.replace(0, pd.NA, inplace=True)
df.dropna(inplace=True)

print(df)