import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average


#
def sigmoide(x):
    return 1 / (1 + np.exp(-x))
#carregamento e formatação dos dados
rawData = open("optdigitsNorm.dat","r").read().splitlines()
rawLabels = open("optdigitsLabels.dat","r").read().splitlines()
entradas = list()
rotulos = list()
meanSquaredErrors =list()
for line in rawData:
    lineFloat = [float(x) for x in line.split(',') if x!='']
    entradas.append(lineFloat)
entradas = np.array(entradas)
for line in rawLabels:
    arr = list()
    for char in line:
        arr.append(np.float64(char))
    rotulos.append(np.array(arr))
rotulos = np.array(rotulos)
#parâmetros de controle
indiceAprendizado = 0.01
nPrevisoesCorretas = 0
epocas = 40
tamanhoC_O=20
tamanhoC_E=len(entradas[0])
tamanhoC_S=len(rotulos[0])
#inicializacao das matrizes
pesosE_O = np.random.uniform(-1, 1, (tamanhoC_O, tamanhoC_E))
biasesE_O = np.zeros((tamanhoC_O, 1))
pesosO_S = np.random.uniform(-1, 1, (tamanhoC_S, tamanhoC_O))
biasesO_S = np.zeros((tamanhoC_S, 1))


for epoca in range(epocas):
    #itera por cada linha e seu respectivo rótulo
    squaredErrors=list()
    for entrada, saidaCorreta in zip(entradas, rotulos):
        #muda o tipo da var entrada de vetor de 5620 posições
        #para uma matriz [64,1], para podermos aplicar operações com numpy
        entrada.shape += (1,)
        saidaCorreta.shape += (1,)
        # Forward propagation entrada -> oculta
        pesosPreAtivacao = biasesE_O + np.dot(pesosE_O,entrada)
        pesosAtivados = sigmoide(pesosPreAtivacao)
        # Forward propagation oculta -> saida
        saidaPreAtivacao = biasesO_S + np.dot(pesosO_S,pesosAtivados)
        saidaObtida = sigmoide(saidaPreAtivacao)
        # Calculo do erro
        e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
        squaredErrors.append(e)
        #checa se previsto=obtido e adiciona ao contador de previsoes
        nPrevisoesCorretas += int(np.argmax(saidaObtida) == np.argmax(saidaCorreta))

        # Backpropagation saida -> oculta (derivada da funcao sigmoide)
        deltaSaida = saidaObtida - saidaCorreta
        pesosO_S += -indiceAprendizado * np.dot(deltaSaida,pesosAtivados.T)
        biasesO_S += -indiceAprendizado * deltaSaida
        # Backpropagation oculta -> entrada (activation function derivative)
        delta_h = np.dot(pesosO_S.T,deltaSaida) * (pesosAtivados * (1 - pesosAtivados))
        pesosE_O += -indiceAprendizado * np.dot(delta_h,entrada.T)
        biasesE_O += -indiceAprendizado * delta_h
    avg = mean(squaredErrors) 
    meanSquaredErrors.append(avg)
    # Show accuracy for this epoch
    print(f"Acc: {round((nPrevisoesCorretas / entradas.shape[0]) * 100, 2)}%")
    nPrevisoesCorretas = 0
plt.plot(meanSquaredErrors)
plt.show()
# Show results
while True:
    index = int(input("Teste um número (0 - 5619): "))
    entrada = entradas[index]
    if index==-1:
        break;
    entrada.shape += (1,)
    # Forward propagation input -> hidden
    pesosPreAtivacao = biasesE_O + pesosE_O @ entrada.reshape(64, 1)
    pesosAtivados = 1 / (1 + np.exp(-pesosPreAtivacao))
    # Forward propagation hidden -> output
    saidaPreAtivacao = biasesO_S + pesosO_S @ pesosAtivados
    saidaObtida = 1 / (1 + np.exp(-saidaPreAtivacao))

    print(f"Desejado: {saidaObtida.argmax()} :)")
    print(rotulos[index])
