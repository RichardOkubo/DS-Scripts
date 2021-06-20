# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:29:01 2017

@author: Jones
"""
import pandas as pd
base = pd.read_csv('credit_data.csv') # carregamento dos dados
"""
base.describe() # descrição dos dados
base.loc[base['age'] < 0] # localida os valores do atributo 'age' que são negativos

base.drop('age', 1, inplace=True) # apagar a coluna
base.drop(base[base.age < 0].index, inplace=True) # ou apagar somente os registros com problema
# ou preencher os valores manualmente / preencher os valores com a média

base.mean() # ver a média de todos os dados
base['age'].mean() # ver somente a média de 'age'
base['age'][base.age > 0].mean() # ver apenas a média do atributo 'age' não-negativos
"""
base.loc[base.age < 0, 'age'] = 40.92 # subistituição dos valores do atributo 'age' negativos para o valor da média encontrada acima
"""
pd.isnull(base['age']) # traz uma tabela com valores booleanos, dizendo se estão preenchidos ou se então vazios, ou seja, 'NaN'
base.loc[pd.isnull(base['age'])] # localiza os valores que não estão preenchidos
"""

# divisão dos valores
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# atribuição de valores para aqueles que estavam vazios
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3]) # formando o shape de 'imputer' com o tamanho de 'previsores'
previsores[:,0:3] = imputer.transform(previsores[:,0:3]) # fazendo a mudança dos dados

# padronização dos dados em uma mesma escala; isso é uma boa prática
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
                  