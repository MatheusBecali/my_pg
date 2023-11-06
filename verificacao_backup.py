# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   E-mail: matheusbecali@gmail.com
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=


# =================== Bibliotecas ===================

from mnist import load_mnist
# import pesos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from glob import glob

import pickle
import os

GRAPHIC_DIR = os.path.join('./Codigo/results_pg')

training_set, test_set = load_mnist(normalize=True, one_hot_label=True)

x_test, y_test = test_set


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


def grafbar_plot(eixo, grafico):
    for i in eixo:
        plt.figure()  # figsize=(5, 5)
        plt.xlabel('Números')
        plt.ylabel('Porcentagem (%)')
        plt.title('Acertos referente ao número {}'.format(i))
        plt.xticks(eixo)  # ajusta a escala do eixo x
        plt.bar(eixo, grafico[i], color="blue")
        plt.show()
        # plt.savefig('Graficos\grafico_{}.png'.format(i))


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

# with open(GRAPHIC_DIR + '/SGD/' + 'pesos_S_300HL_100e.pkl', 'rb') as handle:
#      dataset = pickle.load(handle)

# with open(GRAPHIC_DIR + '/Adam/' + 'pesos_A_300HL_100e.pkl', 'rb') as handle:
#      dataset = pickle.load(handle)

with open(GRAPHIC_DIR + '/SGDM/' + 'pesos_SM_300HL_100e.pkl', 'rb') as handle:
     dataset = pickle.load(handle)

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
# vet_acertos_rmsprop = [100 - vet_acertos_rmsprop[i] for i in range(len(vet_acertos_rmsprop))]

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
# plt.plot(vet_int, vet_acertos_sgd, color='r', linewidth=2, label='SGD')
# plt.plot(vet_int, vet_acertos_sgdm, color='b', linewidth=2, label='SGD_Momentum')
# plt.plot(vet_int, vet_acertos_adagrad, color='m', linewidth=2, label='AdaGrad')
# plt.plot(vet_int, vet_acertos_rmsprop, color='y', linewidth=2, label='RMSProp')
# plt.plot(vet_int, vet_acertos_adam, color='k', linewidth=2, label='Adam')
# plt.xlabel('Epochs ("Iterações")')
# plt.ylabel('Erros (%)')
# plt.yscale('log')

# ax=plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# # plt.xscale('log')
# plt.legend()
# # plt.savefig('Comparação_300HL_CNMAC.pdf', Transparent=True)
# plt.show()


# =================== remanejamento dos arquivos para teste ===================

# esp += [np.argmax(y_test[i]) for i in range(0, len(y_test))]
# matrix += [(x_test[j].reshape(784, 1)) for j in range(0, len(x_test))]
# test_data += [(matrix[k], esp[k]) for k in range(0, len(esp))]
# test_data = list(test_data)

# =================== definicao do vetor a ser treinado ===================

# a = test_data[5][0]
# y = test_data[5][1]

# # =================== feedforward ===================

# a = feedforward(a, b, w)

# test_results = [(np.argmax(a), y)]
# print(test_results)

# # =================== imagem ===================

# vetor, tamanho, tupla, xv = imagem(test_data, b, w)
# print("Erros computados {}".format(tamanho))
# print("Tupla referente ao erro = {}".format(tupla[0]))
# plt.imshow(xv[0].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()

# ================= grafico de barras ===================

# eixo = np.array(range(0, 10, 1))
# grafico, v_qnt = corrf(test_data, b, w)
# grafbar_plot(eixo, grafico)

# ================= matriz de confusão ===================

import pandas as pd

x_results = list()

for x in x_test:
    a = feedforward(x, b, w)
    # print(a)
    x_results.append(a)

print(x_results)
# print(pd.crosstab(y_test, feedfoward()))
# =================== numero no arquivo ===================

# print("Quantidade de cada numero no arquivo de teste:")
# for n in range(0,10):
#     print("{}: {}".format(n,v_qnt[n]))

# =================== Proprios Digitos ===================

img_names = glob(os.path.join('Digitos_Proprios/','*.png'))  # Endereco das imagens
img_names.sort() # Ordem Alfabetica

i_test = own_digits(img_names)  # funcao que processa as imagens e retorna uma tupla

# test_results = list()

# for x,y in i_test:
#     a = feedforward(x, b, w)
#     # print(a)
#     test_results += [(np.argmax(a), y)]

# grafico, v_qnt = corrf(i_test, b, w)

# print("Quantidade de cada numero no arquivo de teste:")
# for n in range(0,10):
#     print("{}: {}".format(n,int(v_qnt[n])))

# ximage = []

# for x,y in i_test:
#     ximage.append(x)

# vetor, tamanho, tupla, xv = imagem(i_test, b, w)
# print("Erros computados {}".format(tamanho))
# print("Tupla referente ao erro = {}".format(tupla[2]))
# print(xv[1].reshape((28, 28)))
# plt.imshow(ximage[10].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()

# =================== Plots Digitos ===================

# fig, axs = plt.subplots(3, 5)
# k=0
# for i in range(0,3):
#     for j in range(0,5):
#         axs[i, j].imshow(ximage[k].reshape((28, 28)), cmap=cm.Greys)
#         k += 2

# #hide axis
# for ax in axs.flat:
#     ax.label_outer()

# #fit axis 0-28
# X_label = [0,28]
# plt.setp(axs, xticks=X_label, yticks=X_label)

# # plt.savefig('Digitos_Proprios2.pdf', Transparent=True)
# plt.show()

# =================== Resultados ===================

test_results = list()

for x,y in i_test:
    a = feedforward(x, b, w)
    # print(a)
    test_results += [(np.argmax(a), y)]

print(test_results)

acertos = sum(int(x == y) for (x, y) in test_results)
porc = acertos / len(test_results)
print("acertos = {}/{}".format(acertos,len(test_results)))
print("porcentagem de acertos = {:2.2%}".format(porc))


# =================== Proprios Digitos (2) ===================

# img = mpimg.imread('Digitos_Proprios/4-1.png')  # altere as imagens que deseja testar
# plt.imshow(img, cmap=cm.Greys_r)
# plt.show()
#
# i_test_y = 4  # caso altere a imagem, altere seu label
# i_test_x = np.resize(img, (28, 28))
# i_test_x = i_test_x.reshape(784, 1)
# i_test = (i_test_x, i_test_y)
# # print(i_test)
#
# x = i_test[0]
# y = i_test[1]

# =================== feedforward ===================

# a = feedforward(x, b, w)
#
# test_results = (np.argmax(a), y)
# print(test_results)

# =================== testes ===================
