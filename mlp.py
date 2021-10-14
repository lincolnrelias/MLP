from math import floor
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax, mean
import seaborn as sn
import pandas as pd
from random import shuffle



#carregamento e formatação dos dados
rawData = open("optdigitsNorm.dat","r").read().splitlines()
rawLabels =[]
arrEntradas = list()
arrSaidasDesejadas = list()
mseTrein =list()
mseValid =list()
mseTeste =list()
#randomiza as posições da linhas dos dados
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
#hiperparâmetros
indiceAprendizado = 0.09
epocas = 0
alpha = .1
neuroniosCO=25
#parâmetros de controle
tamanhoC_E=len(arrEntradas[0])
tamanhoC_S=len(arrSaidasDesejadas[0])
ultAcuTrei = 0
ultAcuValid = 0
ultAcuTeste = 0
#inicializacao das matrizes
pesosE_O = np.random.uniform(-1, 1, (neuroniosCO, tamanhoC_E))
biasesE_O = np.zeros((neuroniosCO, 1))
pesosO_S = np.random.uniform(-1, 1, (tamanhoC_S, neuroniosCO))
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

def treinamento(pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S):
    nPrevisoesCorretas=0
    msErrors=list()
    #itera por cada linha e seu respectivo rótulo
    for entradas, saidaCorreta in zip(entradasTreinamento, saidasTreinamento):
        #muda o tipo da var entradas de vetor de 5620 posições
        #para uma matriz [1,64], para podermos aplicar operações matriciais usando numpy
        entradas.shape += (1,)
        saidaCorreta.shape += (1,)
        # Forward propagation entrada -> oculta
        pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
        # Forward propagation oculta -> saida
        saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)
        # Calculo do erro
        e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
        msErrors.append(e)
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
    ultAcuTrei = round((nPrevisoesCorretas / entradasTreinamento.shape[0]) * 100, 2)
    if epocas%10==0:
        print(f"Acurácia de treinamento: {ultAcuTrei}% , com {epocas} épocas")
    mseTrein.append(mean(msErrors))
    return pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S
    #validacao
def validacao():    
    nPrevisoesCorretas=0
    msErrors=list()
    for entradas, saidaCorreta in zip(entradasValid, saidasValid):
        entradas.shape += (1,)
        saidaCorreta.shape += (1,)
        # Forward propagation entrada -> oculta
        pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
        # Forward propagation oculta -> saida
        saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)
        # Calculo do erro quadrático medio das entradas dessa iteração
        e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
        msErrors.append(e)
        #checa se previsto=obtido e adiciona ao contador de previsoes
        nPrevisoesCorretas += int(np.argmax(saidaObtida) == np.argmax(saidaCorreta))
    ultAcuValid = round((nPrevisoesCorretas / entradasValid.shape[0]) * 100, 2)
    mseValid.append(mean(msErrors))
    if epocas%10==0:
        print(f"Acurácia de validação: {ultAcuValid}%")

# A Simple Confusion Matrix Implementation
def confusionmatrix(actual, predicted):
    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap   = {key: i for i, key in enumerate(unique)}
    # Generate Confusion Matrix
    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1
    return matrix

def teste(gerarMatrizConfusao):    
    nPrevisoesCorretas=0
    msErrors=list()
    valPrevistos = list()
    valObtidos = list()
    for entradas, saidaCorreta in zip(entradasTeste, saidasTeste):
        entradas.shape += (1,)
        saidaCorreta.shape += (1,)
        # Forward propagation entrada -> oculta
        pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
        # Forward propagation oculta -> saida
        saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)
        # Calculo do erro quadrático medio das entradas dessa iteração
        e = 1 / len(saidaObtida) * np.sum((saidaObtida - saidaCorreta) ** 2, axis=0)
        msErrors.append(e)
        #checa se previsto=obtido e adiciona ao contador de previsoes
        valPrevistos.append(np.argmax(saidaCorreta))
        valObtidos.append(np.argmax(saidaObtida))
        nPrevisoesCorretas += int(np.argmax(saidaObtida) == np.argmax(saidaCorreta))
    ultAcuTeste = round((nPrevisoesCorretas / entradasTeste.shape[0]) * 100, 2)
    mseTeste.append(mean(msErrors))
    if epocas%10==0 and not gerarMatrizConfusao:
        print(f"Acurácia de teste: {ultAcuTeste}%")
    if gerarMatrizConfusao:
        cf = confusionmatrix(valObtidos,valPrevistos)

        sn.heatmap(cf, annot=True)
        plt.xlabel("Valores Obtidos")
        plt.ylabel("Valores Previstos")
        plt.show()

while (ultAcuValid<97 or ultAcuTrei<99) and epocas<50:
    epocas+=1
    pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S = treinamento(
        pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S)
    validacao()
    teste(False)
teste(True)


# Show results
plt.plot(mseTrein)
plt.plot(mseValid)
plt.plot(mseTeste)
plt.show()
while True:
    index = int(input("Teste um número (0 - 5619): "))
    entradas = arrEntradas[index]
    if index==-1:
        break;
    entradas.shape += (1,)
    # Forward propagation entrada -> oculta
    pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
    # Forward propagation oculta -> saida
    saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)

    print(f"Desejado: {index}")
    print(f"Obtido: {saidaObtida.argmax()}")
