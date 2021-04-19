import numpy as np


def sigmoidDerivada(sig):
	return sig * (1 - sig)


def sigmoid(soma):
	return 1 / (1 + np.exp(-soma)) #fórmula da função Sigmoidal


'''
a = sigmoid = (-1.5) # exemplo
b = np.exp(0) # exemplo

c = sigmoid(0.5) # exemplo
d = sigmoidDerivada(c) # exemplo
'''

# Cada registro possue duas entradas
entradas = np.array([[0, 0],  # 1º registro
					 [0, 1],  # 2º registro
					 [1, 0],  # 3º registro
					 [1, 1]]) # 4º registro

saidas = np.array([[0], [1], [1], [0]]) # Operador XOR

'''
# Sinapse da Camada de Entrada para a Camada Oculta
pesos0 = np.array([[-0.424, -0.740, -0.961], # Pesos da 1ª entrada [ou seja, x1]
                   [0.358, -0.577, -0.469]]) # Pesos do 2ª entrada [ou seja, x2]
    
# Sinapse da Camada Oculta para a Camada de Saída
pesos1 = np.array([[-0.017], [-0.893], [0.148]])
'''

# Inicialização dos pesos aleatóriamente
pesos0 = 2 * np.random.random((2, 3)) - 1 # dois neurônios na camada de entrada e 3 neurônios na camada oculta
pesos1 = 2 * np.random.random((3, 1)) # três neurônios na camada oculta e apenas um neurônios na camada de saída

epocas = 1000000 # exemplo
taxaAprendizagem = 0.5 # exemplo
momento = 1 # exemplo, neste caso, é neutro 

for j in range(epocas):
	camadaEntrada = entradas # cópia
	somaSinapse0 = np.dot(camadaEntrada, pesos0) # Entrada para Oculta
	camadaOculta = sigmoid(somaSinapse0)

	somaSinapse1 = np.dot(camadaOculta, pesos1) # Oculta para Saída
	camadaSaida = sigmoid(somaSinapse1)

	erroCamadaSaida = saidas - camadaSaida
	mediaAbsoluta = np.mean(np.abs(erroCamadaSaida)) # média absoluta (ou seja, não negativa) dos valores
	print("Erro: " + str(mediaAbsoluta))

	derivadaSaida = sigmoidDerivada(camadaSaida)
	deltaSaida = erroCamadaSaida * derivadaSaida # fórmula do delta para a Camada de Saída

	pesos1Transposta = pesos1.T # '.T' -> matriz transposta para funcionar a multiplicação abaixo
	deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta) # '.dot' é uma multiplicação escalar
	deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

	camadaOcultaTransposta = camadaOculta.T # matriz transposta
	# Backpropagation / Retropropagação
	pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida) # atualização dos pesos
	pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem) # fórmula do Back...

	camadaEntradaTransposta = camadaEntrada.T # matriz transposta
	# Backpropagation / Retropropagação
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)  # atualização dos pesos
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem) # fórmula do Back...
