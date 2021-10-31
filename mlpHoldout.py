from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from random import shuffle
import sys
import os



#carregamento e formatação dos dados
rawData = open(sys.argv[1], "r").read().splitlines()
try:
    os.mkdir("saidas")
except OSError as e:
    pass;

outputFile = open("saidas\\"+sys.argv[2],'w')
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
        # Backpropagation oculta -> entrada (com derivada da sigmóide)
        delta_h = np.dot(pesosO_S.T,deltaS_O) * (pesosAtivados * (1 - pesosAtivados))
        pesosE_O += -indiceAprendizado * np.dot(delta_h,entradas.T)+alpha*lastDeltaE_O
        lastDeltaE_O = -indiceAprendizado * np.dot(delta_h,entradas.T)
        biasesE_O += -indiceAprendizado * delta_h
    #Acurácia para essa época
    _ultAcuTrei = round((nPrevisoesCorretas / entradasTreinamento.shape[0]) * 100, 2)
    if epocas%10==0:
        print(f"Acurácia de treinamento: {ultAcuTrei}% , com {epocas} épocas")
    mseTrein.append(np.mean(msErrors))
    return pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S,_ultAcuTrei
    
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
    mseValid.append(np.mean(msErrors))
    if epocas%10==0:
        print(f"Acurácia de validação: {ultAcuValid}%")
    return round((nPrevisoesCorretas / entradasValid.shape[0]) * 100, 2)

def confusionmatrix(obtido, previsto):
    unico = sorted(set(obtido))
    matriz = [[0 for x in unico] for x in unico]
    mapa   = {chave: i for i, chave in enumerate(unico)}
    #Gera a matriz de confusao
    for p, o in zip(previsto, obtido):
        matriz[mapa[p]][mapa[o]] += 1
    return matriz

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
    mseTeste.append(np.mean(msErrors))
    if epocas%10==0 and not gerarMatrizConfusao:
        print(f"Acurácia de teste: {ultAcuTeste}%")
    if gerarMatrizConfusao:
        cf = confusionmatrix(valObtidos,valPrevistos)

        sn.heatmap(cf, annot=True)
        plt.xlabel("Valores Obtidos")
        plt.ylabel("Valores Previstos")
        fig = plt.gcf()
        fig.set_size_inches(16.5, 10.5)
        plt.savefig("saidas\\matriz_confusao.png",dpi=200)
        plt.clf()
    return ultAcuTeste
print("Executando treinamento e validação da rede:")
while ultAcuValid<=95 and ultAcuTrei<=99:
    epocas+=1
    pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S,ultAcuTrei = treinamento(
        pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S)
    ultAcuValid = validacao()
    ultAcuTeste=teste(False)
print("ÉPOCA DE CONVERGÊNCIA(95%% de acurácia no conjunto de validação): ",epocas)
print("Acurácia de treinamento: ",ultAcuTrei,"%%")
print("Acurácia de validação: ",ultAcuValid,"%%")
print("Acurácia de teste: ",ultAcuTeste,"%%")
teste(True)


#gráficos curvas de aprendizado
red = mpatches.Patch(color='red', label='Treinamento')
yellow = mpatches.Patch(color='green', label='Validação')
blue = mpatches.Patch(color='blue', label='Teste')
fig, axs = plt.subplots(3)
axs[0].plot(mseTrein,color="red",linewidth=4)
axs[0].set_title('Treinamento')
axs[1].plot(mseValid,color="green",linewidth=4)
axs[1].set_title('Validação')
axs[2].plot(mseTeste,color="blue",linewidth=4)
axs[2].set_title('Teste')
for ax in axs.flat:
    ax.set(xlabel='Épocas', ylabel='Erros médios')
for ax in axs.flat:
    ax.label_outer()
fig = plt.gcf()
fig.set_size_inches(14.5, 18.5)
plt.savefig("saidas\\curvas_aprendizado.png",dpi=200)
#escrita de erros no arquivo de saida
outputFile.write("Erros quadraticos:\n")
outputFile.write("Epoca;Treinamento;Validacao,Teste\n")
outputText = list()
for x in range(len(mseTrein)):
    outputText.append(str(x+1)+";"+str(mseTrein[x])+";"+str(mseValid[x])+";"+str(mseTeste[x])+"\n")
outputFile.writelines(outputText)
erroMedio = sum(mseTeste)/50
erroVerdadeiro = np.sqrt((erroMedio*(1-erroMedio))/5620)
outputFile.write("Epocas para convergencia: "+str(epocas)+"; Erro verdadeiro: "+str(erroVerdadeiro))
outputFile.close()
print("Resultados salvos na pasta 'saídas'.")
#usado para testar valores individuais:
#while True:
#    index = int(input("Teste um numero (0 - 5619): "))
#    entradas = arrEntradas[index]
#    if index==-1:
#        break;
#    entradas.shape += (1,)
#    # Forward propagation entrada -> oculta
#    pesosAtivados = forward_propagate(pesosE_O,entradas,biasesE_O)
    # Forward propagation oculta -> saida
#   saidaObtida = forward_propagate(pesosO_S,pesosAtivados,biasesO_S)

#    print(f"Desejado: {arrSaidasDesejadas[index].argmax()}")
#    print(f"Obtido: {saidaObtida.argmax()}")
