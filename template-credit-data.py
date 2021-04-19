# -*- coding: utf-8 -*-
import pandas as pd

base = pd.read_csv('credit_data.csv') # Carregamento da base de dados
base.loc[base.age < 0, 'age'] = 40.92 # Tratamento dos dados inconsistentes

# Divisão dos dados em 'amostra' e 'alvos'
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Tratamento dos dados faltantes
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
'''
imputer = imputer.fit(previsores[:, 1:4]) # errado?
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4]) # errado?
'''
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# Escalonamento dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)            

# Divisão da 'amostra' e do 'alvo' em dados_treino e dados_teste
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Importação da biblioteca (ou seja, o algoritmo de Machine Learning escolhido)
# ... from sklearn.naive_bayes import GaussianNB
# Criação do classificador
# ... classificador = GaussianNB()

classificador.fit(previsores_treinamento, classe_treinamento) # aqui é efetivamente realizado o treinamento
previsoes = classificador.predict(previsores_teste) # aqui é mostrado os resultados dos testes

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes) # percentual de acerto
#print(precisao)
matriz = confusion_matrix(classe_teste, previsoes) # matriz que apresenta os erros e acertos
#print(matriz)
