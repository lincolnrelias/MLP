import numpy as np
def sigmoide(x):
    return 1/(1+np.exp(-x))

def forward(entradas):
    inputsAtivados = entradas
    for linhaPesos in matrizPesos:
        netInputs = np.dot(inputsAtivados,linhaPesos)
        inputsAtivados = sigmoide(netInputs)
    return inputsAtivados
#carrega os dados
dadosDigitos = open("optdigitsNorm.dat","r").read().splitlines()
listaDados = list()
for line in dadosDigitos:
    lineSplit = line.split(',')
    lineSplit.pop()
    lineSplitFloat = [float(x) for x in lineSplit]
    listaDados.append(lineSplitFloat)
resultadosEsperados = open("optdigitisRotulos.dat","r").read().splitlines()
resultadosEsperados = [int(x) for x in resultadosEsperados]

#inicializa os conjuntos de dados
tercoDosDados = int(np.floor(len(listaDados)/3))
conjuntoTreinamento = [listaDados[x] for x in range(tercoDosDados)]
conjuntoValidacao = [listaDados[x] for x in range(tercoDosDados,tercoDosDados*2)]
conjuntoTeste = [listaDados[x] for x in range(tercoDosDados*2,tercoDosDados*3)]
neuroniosCamadaOculta=5
neuroniosSaida=10
neuroniosInput=len(listaDados[0])
indiceAprendizado=0.1
matrizPesos = []
matrizPesos.append(np.random.rand((neuroniosInput,neuroniosCamadaOculta)))
matrizPesos.append(np.random.rand((neuroniosCamadaOculta,neuroniosSaida)))

#teste
input = np.array([conjuntoTeste])
target = np.array([7])
a = forward(input)
print()
