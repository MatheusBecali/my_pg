# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   Matricula: 2017101659
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=


# =================== Bibliotecas ===================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

import pickle
import os

GRAPHIC_DIR = os.path.join('E:\Dropbox\DeepLearningPython35-master\RedeNeural_Mnist\Codigo\Relatorio_Final')


# =================== Funcoes ===================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def feedforward(a, b, w):
    b1, b2 = b
    w1, w2 = w
    x1 = np.dot(w1, a) + b1
    a1 = sigmoid(x1)
    x2 = np.dot(w2, a1) + b2
    a2 = sigmoid(x2)

    return a2


def corrf(test_data, b, w):
    qnt_k = np.zeros(10)
    grafico = np.zeros((10, 10))

    for (x, y) in test_data:
        activation = feedforward(x, b, w)  # Executa o feedforward na rede treinada
        qnt_k[y] += 1  # Quantidade iamgens de cada numeros, onde k = 0, 1, ..., 9
        for i in range(0, 10):
            grafico[y][i] += activation[i]

    for i in range(10):
        for j in range(0, 10):
            grafico[i][j] = (100 * grafico[i][j]) / qnt_k[i]

    return grafico, qnt_k


def imagem(test_data, b, w):
    i, tam_err = 0, 0
    vetor = list()
    tupla = list()
    x_err = list()

    for x, y in test_data:
        x_tr = np.argmax(feedforward(x, b, w))
        if int(x_tr != y):
            vetor.append(i)  # Armazena a Posicao da tupla no test_data
            tupla.append((x_tr, y))  # Armazena a tupla a qual o X != Y
            x_err.append(x)  # Armazena o vetor referente a imagem de X errada
            i += 1
            tam_err += 1  # tamanho do erro
        else:
            i += 1

    return vetor, tam_err, tupla, x_err

def own_digits(img_names):
    img = list()
    i_test_y = list()
    i_test_x = list()

    img += [mpimg.imread(fn) for fn in img_names]

    for i in range(0, len(img_names)):
        i_test_x.append(np.resize(img[i], (28, 28)))
        i_test_x[i] = i_test_x[i].reshape(784, 1)
        aux = img_names[i].split('Digitos_Proprios\\') #Windows
        # aux = img_names[i].split('Digitos_Proprios/') #Linux
        aux_2 = aux[1].split('.png')
        aux_3 = aux_2[0].split('-')
        i_test_y.append(int(aux_3[0]))

    return zip(i_test_x, i_test_y)

def graffeed_plot(vet_acertos, vet_int):
    plt.plot(vet_int, vet_acertos, color='r', linewidth=2, label='SGD')
    plt.xlabel('Epochs')
    plt.ylabel('Acertos (%)')
    # plt.title('SGD -> hl=200, E=200, lr=0.001')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.savefig('Grafico_SGD.png', Transparent=True)
    plt.show()

# =================== definicao variaveis ===================

esp = list()
matrix = list()
test_data = list()

# =================== chamando o arquivo pesos.py ===================

with open(GRAPHIC_DIR + '/SGD/' + 'pesos_S_300HL_100e.pkl', 'rb') as handle:
     dataset = pickle.load(handle)

# with open(GRAPHIC_DIR + '/Adam/' + 'pesos_A_300HL_100e.pkl', 'rb') as handle:
#      dataset = pickle.load(handle)

w = (dataset['weights_1'], dataset['weights_2'])
b = (dataset['biases_1'], dataset['biases_2'])

# w = pesos.weights()
# b = pesos.biases()

# =================== chamando o arquivo axis.py ===================

# with open('axis.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos = axis['y']
# vet_acertos = [100 - vet_acertos[i] for i in range(len(vet_acertos))]

# graffeed_plot(vet_acertos, vet_int)

# =================== Grafico Geral ===================

# with open(GRAPHIC_DIR + '/SGD/' + 'axis_S_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos_sgd = axis['y']
# vet_acertos_sgd = [100 - vet_acertos_sgd[i] for i in range(len(vet_acertos_sgd))]

# with open(GRAPHIC_DIR + '/SGDM/' + 'axis_SM_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_sgdm = axis['y']
# vet_acertos_sgdm = [100 - vet_acertos_sgdm[i] for i in range(len(vet_acertos_sgdm))]

# with open(GRAPHIC_DIR + '/AdaGrad/' + 'axis_AG_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_adagrad = axis['y']
# vet_acertos_adagrad = [100 - vet_acertos_adagrad[i] for i in range(len(vet_acertos_adagrad))]

# with open(GRAPHIC_DIR + '/RMSProp/' + 'axis_RP_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_rmsprop = axis['y']
# vet_acertos_rmsprop = [100 -  vet_acertos_rmsprop[i] for i in range(len(vet_acertos_rmsprop))]

# with open(GRAPHIC_DIR + '/Adam/' + 'axis_A_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_adam = axis['y']
# vet_acertos_adam = [100 - vet_acertos_adam[i] for i in range(len(vet_acertos_adam))]

# =================== Graficos ===================

# with open(GRAPHIC_DIR + '/SGD/' + 'axis_S_100HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos_sgd = axis['y']
# vet_acertos_sgd = [100 - vet_acertos_sgd[i] for i in range(len(vet_acertos_sgd))]

# with open(GRAPHIC_DIR + '/SGD/' + 'axis_S_200HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos_sgd2 = axis['y']
# vet_acertos_sgd2 = [100 - vet_acertos_sgd2[i] for i in range(len(vet_acertos_sgd2))]

# with open(GRAPHIC_DIR + '/SGD/' + 'axis_S_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos_sgd3 = axis['y']
# vet_acertos_sgd3 = [100 - vet_acertos_sgd3[i] for i in range(len(vet_acertos_sgd3))]

# with open(GRAPHIC_DIR + '/Adam/' + 'axis_A_100HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_int = axis['x']
# vet_acertos_adam = axis['y']
# vet_acertos_adam = [100 - vet_acertos_adam[i] for i in range(len(vet_acertos_adam))]

# with open(GRAPHIC_DIR + '/Adam/' + 'axis_A_200HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_adam2 = axis['y']
# vet_acertos_adam2 = [100 - vet_acertos_adam2[i] for i in range(len(vet_acertos_adam2))]

# with open(GRAPHIC_DIR + '/Adam/' + 'axis_A_300HL_100e.pkl', 'rb') as handle:
#     axis = pickle.load(handle)

# vet_acertos_adam3 = axis['y']
# vet_acertos_adam3 = [100 - vet_acertos_adam3[i] for i in range(len(vet_acertos_adam3))]
###
# plt.xticks(vet_int)
###
# plt.plot(vet_int, vet_acertos_sgd3, color='r', linewidth=2, label='SGD_300hl')
# plt.plot(vet_int, vet_acertos_sgd2, color='b', linewidth=2, label='SGD_200hl')
# plt.plot(vet_int, vet_acertos_sgd, color='m', linewidth=2, label='SGD_100hl')
# plt.plot(vet_int, vet_acertos_adam3, color='k', linewidth=2, label='Adam_300hl')
# plt.plot(vet_int, vet_acertos_adam2, color='y', linewidth=2, label='Adam_200hl')
# plt.plot(vet_int, vet_acertos_adam, color='pink', linewidth=2, label='Adam_100hl')
###
# fig = plt.figure(figsize=(10,6))
# plt.plot(vet_int, vet_acertos_sgd, color='#1F77B4', linewidth=1.5, label='SGD')
# plt.plot(vet_int, vet_acertos_sgdm, color='#FF7F0E', linewidth=1.5, label='SGD_Momentum')
# plt.plot(vet_int, vet_acertos_adagrad, color='#2CA02C', linewidth=1.5, label='AdaGrad')
# plt.plot(vet_int, vet_acertos_rmsprop, color='#D62728', linewidth=1.5, label='RMSProp')
# plt.plot(vet_int, vet_acertos_adam, color='#9467BD', linewidth=1.5, label='Adam')
# plt.xlabel('Epochs ("Iterações")')
# plt.ylabel('Erros (%)')
# plt.yscale('log')
# # plt.xscale('log')
# plt.legend()
# # plt.savefig('Comparação_300HL_novo.pdf', Transparent=True)
# plt.show()

# =================== Proprios Digitos ===================

img_names = glob(os.path.join('Digitos_Proprios/','*.png'))  # Endereco das imagens
img_names.sort() # Ordem Alfabetica

i_test = own_digits(img_names)  # funcao que processa as imagens e retorna uma tupla

# vetor, tamanho, tupla, xv = imagem(i_test, b, w)
# print("Erros computados {}".format(tamanho))
# print("Tupla referente ao erro = {}".format(tupla[2]))
# # print(xv[1].reshape((28, 28)))
# plt.imshow(xv[2].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()

test_results = list()

for x, y in i_test:
    a = feedforward(x, b, w)
    # print(a)
    test_results += [(np.argmax(a), y)]
print(test_results)

acertos = sum(int(x == y) for (x, y) in test_results)
porc = acertos / len(test_results)
print("acertos = {}/{}".format(acertos,len(test_results)))
print("porcentagem de acertos = {:2.2%}".format(porc))
