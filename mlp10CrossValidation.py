from math import floor, sqrt
import numpy as np
from random import shuffle
import sys



#carregamento e formatação dos dados
rawData = open(sys.argv[1], "r").read().splitlines()

outputFile = open(sys.argv[2],'w')
rawLabels =[]
arrEntradas = list()
arrSaidasDesejadas = list()

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
tamFold = floor(len(arrEntradas)/10)
mseTrein =list()
mseValid =list()
mseTeste =list()
entradasTreinamento=[]
entradasTeste=[]
saidasTreinamento=[]
saidasTeste=[]
erros = list()
#hiperparâmetros
indiceAprendizado = 0.09
epocas = 15
alpha = .1
neuroniosCO=25
#parâmetros de controle
tamanhoC_E=len(arrEntradas[0])
tamanhoC_S=len(arrSaidasDesejadas[0])
ultAcuTrei = 0
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
    _ultAcuTrei = round((nPrevisoesCorretas / entradasTreinamento.shape[0]) * 100, 2)
    mseTrein.append(np.mean(msErrors))
    return pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S,_ultAcuTrei
    

def teste():    
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
    return ultAcuTeste
print("Executando treinamento e validação da rede MLP por metodo 10 fold cross-validation:")
outputFile.write("Fold;Erro Verdadeiro\n")
for i in range(10):
    if i>0:
        entradasTreinamento = arrEntradas[0:tamFold*(i-1)]
        saidasTreinamento = arrSaidasDesejadas[0:tamFold*(i-1)]
    if i<9:
        if i>0:
            entradasTreinamento=np.concatenate((entradasTreinamento,arrEntradas[tamFold*(i+1):tamFold*10]),0)
            saidasTreinamento= np.concatenate((saidasTreinamento,arrSaidasDesejadas[tamFold*(i+1):tamFold*10]),0)
        else:
            entradasTreinamento= arrEntradas[tamFold*(i+1):tamFold*10]
            saidasTreinamento= arrSaidasDesejadas[tamFold*(i+1):tamFold*10]
    entradasTeste = arrEntradas[tamFold*i:tamFold*(i+1)] 
    saidasTeste = arrSaidasDesejadas[tamFold*i:tamFold*(i+1)]
    print("Treinando e testando com fold teste: ",i+1)
    for j in range(epocas):
        pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S,ultAcuTrei = treinamento(
            pesosE_O,biasesE_O,lastDeltaE_O,pesosO_S,biasesO_S,lastDeltaO_S)
        ultAcuTeste=teste()
    print("Acurácias para fold ",i+1)
    print("Treinamento: ",ultAcuTrei," Teste:",ultAcuTeste)
    pesosE_O = np.random.uniform(-1, 1, (neuroniosCO, tamanhoC_E))
    biasesE_O = np.zeros((neuroniosCO, 1))
    pesosO_S = np.random.uniform(-1, 1, (tamanhoC_S, neuroniosCO))
    biasesO_S = np.zeros((tamanhoC_S, 1))
    lastDeltaE_O = 0
    lastDeltaO_S = 0
    erroMedio = sum(mseTeste)/50
    erroVerdadeiro = np.sqrt((erroMedio*(1-erroMedio))/5620)
    erros.append(100-ultAcuTeste)
    outputFile.write(str(i+1)+";"+str(erroVerdadeiro)+"\n")
    mseTeste=list()

#escrita de erros no arquivo de saida

outputFile.write("Considerando a taxa de erro medio= "+str(round((sum(erros)/10),2))+chr(37)+" \n")
outputFile.write("e levando em consideracao a quantidade de exemplos = 5620\n")
e = sum(erros)/1000
erroModelo=np.sqrt((e*(1-e))/5620)

outputFile.write("Com 95"+chr(37)+" de confianca, temos o termo de desempenho do modelo entre "
+str(100-(round(e-1.96*erroModelo,3)*100))+chr(37)+" e "
+str(100-(round(e+1.96*erroModelo,3)*100))+chr(37))
outputFile.close()
print("Resultados salvos no arquivo de saida.")
