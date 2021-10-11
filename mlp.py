from math import floor
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from random import shuffle



#carregamento e formatação dos dados
rawData = open("optdigitsNorm.dat","r").read().splitlines()
rawLabels =[]
arrEntradas = list()
arrSaidasDesejadas = list()
mseTrein =list()
mseValid =list()
mseTeste =list()
shuffle(rawData)
for line in rawData:
    splitLine = line.split(',')
    lineFloat = [float(splitLine[x]) for x in range(64)]
    rawLabels.append(splitLine[64])
    arrEntradas.append(lineFloat)
arrEntradas = np.array(arrEntradas)
for line in rawLabels:
    arr = list()
    for char in line:
        arr.append(np.float64(char))
    arrSaidasDesejadas.append(np.array(arr))
arrSaidasDesejadas = np.array(arrSaidasDesejadas)
tercoSaidas = floor(len(arrEntradas)/3)
entradasTreinamento = arrEntradas[0:tercoSaidas]
entradasValid = arrEntradas[tercoSaidas:2*tercoSaidas]
entradasTeste = arrEntradas[tercoSaidas*2:tercoSaidas*3]
saidasTreinamento = arrSaidasDesejadas[0:tercoSaidas]
saidasValid = arrSaidasDesejadas[tercoSaidas:2*tercoSaidas]
saidasTeste = arrSaidasDesejadas[tercoSaidas*2:tercoSaidas*3]
#parâmetros de controle
indiceAprendizado = 0.1
nPrevisoesCorretas = 0
epocas = 70
alpha = 0.5
tamanhoC_O=20
tamanhoC_E=len(arrEntradas[0])
tamanhoC_S=len(arrSaidasDesejadas[0])
ultPrecTrei = 0
ultPrecValid = 0
ultPrecTeste = 0
#inicializacao das matrizes
pesosE_O = np.random.uniform(-1, 1, (tamanhoC_O, tamanhoC_E))
biasesE_O = np.zeros((tamanhoC_O, 1))
pesosO_S = np.random.uniform(-1, 1, (tamanhoC_S, tamanhoC_O))
biasesO_S = np.zeros((tamanhoC_S, 1))
lastDeltaE_O = 0
lastDeltaO_S = 0

def sigmoide(x):
    return 1 / (1 + np.exp(-x))
def forward_propagate(pesos,valores,biases):
    # Forward propagation entrada -> oculta
    somatoria = np.dot(pesos,valores) + biases
    valoresPosAtivacao = sigmoide(somatoria)
    return valoresPosAtivacao

for epoca in range(epocas):
    #itera por cada linha e seu respectivo rótulo
    squaredErrors=list()
    for entradas, saidaCorreta in zip(entradasTreinamento, saidasTreinamento):
        #muda o tipo da var entrada de vetor de 5620 posições
        #para uma matriz [64,1], para podermos aplicar operações com numpy
        entradas.shape += (1,)
        saidaCorreta.shape += (1,)
        # Forward propagation entrada -> oculta
        pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
        # Forward propagation oculta -> saida
        saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)
        # Calculo do erro
        e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
        mseTrein.append(e)
        #checa se previsto=obtido e adiciona ao contador de previsoes
        nPrevisoesCorretas += int(np.argmax(saidaObtida) == np.argmax(saidaCorreta))

        # Backpropagation saida -> oculta (derivada da funcao sigmoide)
        deltaS_O = saidaObtida - saidaCorreta
        
        pesosO_S += -indiceAprendizado * np.dot(deltaS_O,pesosAtivados.T) + alpha*lastDeltaO_S
        lastDeltaO_S =  -indiceAprendizado * np.dot(deltaS_O,pesosAtivados.T)
        biasesO_S += -indiceAprendizado * deltaS_O
        # Backpropagation oculta -> entrada (activation function derivative)
        delta_h = np.dot(pesosO_S.T,deltaS_O) * (pesosAtivados * (1 - pesosAtivados))
        pesosE_O += -indiceAprendizado * np.dot(delta_h,entradas.T)+alpha*lastDeltaE_O
        lastDeltaE_O = -indiceAprendizado * np.dot(delta_h,entradas.T)
        biasesE_O += -indiceAprendizado * delta_h
    # Show accuracy for this epoch
    ultPrecTrei = round((nPrevisoesCorretas / entradasTreinamento.shape[0]) * 100, 2)
    print(f"Precisão de treinamento: {ultPrecTrei}%")
    nPrevisoesCorretas = 0

#validacao
for entradas, saidaCorreta in zip(entradasValid, saidasValid):
    entradas.shape += (1,)
    saidaCorreta.shape += (1,)
    # Forward propagation entrada -> oculta
    pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
    # Forward propagation oculta -> saida
    saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)
        # Calculo do erro
    e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
    mseValid.append(e)
    #checa se previsto=obtido e adiciona ao contador de previsoes
    nPrevisoesCorretas += int(np.argmax(saidaObtida) == np.argmax(saidaCorreta))
ultPrecValid = round((nPrevisoesCorretas / entradasValid.shape[0]) * 100, 2)
print(f"Precisão de validação: {ultPrecValid}%")
# Show results
plt.plot(mseTrein)
plt.plot(mseValid)
plt.show()
while True:
    index = int(input("Teste um número (0 - 5619): "))
    entradas = arrEntradas[index]
    if index==-1:
        break;
    entradas.shape += (1,)
    # Forward propagation input -> hidden
    pesosPreAtivacao = biasesE_O + pesosE_O @ entradas.reshape(64, 1)
    pesosAtivados = 1 / (1 + np.exp(-pesosPreAtivacao))
    # Forward propagation hidden -> output
    saidaPreAtivacao = biasesO_S + pesosO_S @ pesosAtivados
    saidaObtida = 1 / (1 + np.exp(-saidaPreAtivacao))

    print(f"Desejado: {saidaObtida.argmax()} :)")
    print(arrSaidasDesejadas[index])
