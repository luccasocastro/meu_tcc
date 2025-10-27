import yfinance as yf
import pandas as pd


def calcular_valorizacao(acoes, data_inicio, data_fim):
    """
    Calcula a valorização percentual de uma lista de ações entre duas datas.

    Parâmetros:
        acoes (list): Lista de tickers das ações (ex: ['PETR4.SA', 'VALE3.SA']).
        data_inicio (str): Data de início no formato 'YYYY-MM-DD'.
        data_fim (str): Data de fim no formato 'YYYY-MM-DD'.

    Retorna:
        DataFrame com a valorização percentual de cada ação.
    """
    dados = yf.download(acoes, start=data_inicio, end=data_fim)['Close']

    valorizacao = {}
    for acao in acoes:
        preco_inicial = dados[acao].iloc[0]
        preco_final = dados[acao].iloc[-1]
        variacao_percentual = (
            (preco_final - preco_inicial) / preco_inicial) * 100
        valorizacao[acao] = variacao_percentual

    df = pd.DataFrame.from_dict(
        valorizacao, orient='index', columns=['Valorização (%)'])
    return df.round(2)


# Exemplo de uso
if __name__ == "__main__":
    acoes = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']

    data_inicio = '2020-06-08'
    data_fim = '2025-06-08'

    resultado = calcular_valorizacao(acoes, data_inicio, data_fim)
    print(resultado)
