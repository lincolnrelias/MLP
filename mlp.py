import numpy as np
import matplotlib.pyplot as plt


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
biasesE_O = np.zeros((20, 1))
pesosO_S = np.random.uniform(-1, 1, (tamanhoC_S, tamanhoC_O))
biasesO_S = np.zeros((10, 1))


for epoca in range(epocas):
    #itera por cada linha e seu respectivo rótulo
    for entrada, rotulo in zip(entradas, rotulos):
        #muda o tipo da entrada de vetor de 5620 posições
        #para uma matriz [64,1], para podermos aplicar operações com numpy
        entrada.shape += (1,)
        rotulo.shape += (1,)
        # Forward propagation input -> hidden
        pesosPreAtivacao = biasesE_O + np.dot(pesosE_O,entrada)
        pesosAtivados = sigmoide(pesosPreAtivacao)
        # Forward propagation hidden -> output
        saidaPreAtivacao = biasesO_S + np.dot(pesosO_S,pesosAtivados)
        saidaAtivada = sigmoide(saidaPreAtivacao)
        # Calculo do erro
        e = 1 / len(saidaAtivada) * np.sum((saidaAtivada - rotulo) ** 2, axis=0)
        #checa se previsto=obtido e adiciona ao contador de previsoes
        nPrevisoesCorretas += int(np.argmax(saidaAtivada) == np.argmax(rotulo))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = saidaAtivada - rotulo
        pesosO_S += -indiceAprendizado * np.dot(delta_o,pesosAtivados.T)
        biasesO_S += -indiceAprendizado * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.dot(pesosO_S.T,delta_o) * (pesosAtivados * (1 - pesosAtivados))
        pesosE_O += -indiceAprendizado * np.dot(delta_h,entrada.T)
        biasesE_O += -indiceAprendizado * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nPrevisoesCorretas / entradas.shape[0]) * 100, 2)}%")
    meanSquaredErrors.append(round((nPrevisoesCorretas / entradas.shape[0]) * 100, 2))
    nPrevisoesCorretas = 0
plt.plot(meanSquaredErrors)
plt.show()
# Show results
while True:
    index = int(input("Teste um número (0 - 5619): "))
    entrada = entradas[index]

    entrada.shape += (1,)
    # Forward propagation input -> hidden
    pesosPreAtivacao = biasesE_O + pesosE_O @ entrada.reshape(64, 1)
    pesosAtivados = 1 / (1 + np.exp(-pesosPreAtivacao))
    # Forward propagation hidden -> output
    saidaPreAtivacao = biasesO_S + pesosO_S @ pesosAtivados
    saidaAtivada = 1 / (1 + np.exp(-saidaPreAtivacao))

    print(f"Desejado: {saidaAtivada.argmax()} :)")
    print(rotulos[index])
