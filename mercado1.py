import pandas as pd

dados = pd.read_csv('mercado.csv', header = None)
transacoes = []
for i in range(0, 10):
    transacoes.append([str(dados.values[i,j]) for j in range(0, 4)])

from apyori import apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_length = 2)

resultados = list(regras)
resultados

myResults = [list(x) for x in resultados]
myRes = []
for j in range(0, 3):
    myRes.append([list(x) for x in myResults[j][2]])
myRes
