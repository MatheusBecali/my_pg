# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   Matricula: 2017101659
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=


# =================== Bibliotecas ===================

from mnist import load_mnist
import pesos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from glob import glob
import pickle
import os

# ============= Importacao dos Dados =============

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


def graf_plot(eixo, grafico):
    for i in eixo:
        plt.figure()  # figsize=(5, 5)
        plt.xlabel('Numeros')
        plt.ylabel('Porcentagem (%)')
        plt.title('Acertos referente ao numero {}'.format(i))
        plt.xticks(eixo)  # ajusta a escala do eixo x
        plt.bar(eixo, grafico[i], color="blue")
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


# =================== definicao variaveis ===================

esp = list()
matrix = list()
test_data = list()

# =================== chamando o arquivo pesos.py ===================

with open('pesos.pkl', 'rb') as handle:
    dataset = pickle.load(handle)

w = (dataset['weights_1'], dataset['weights_2'])
b = (dataset['biases_1'], dataset['biases_2'])

# w = pesos.weights()
# b = pesos.biases()

# =================== remanejamento dos arquivos para teste ===================

esp += [np.argmax(y_test[i]) for i in range(0, len(y_test))]
matrix += [(x_test[j].reshape(784, 1)) for j in range(0, len(x_test))]
test_data += [(matrix[k], esp[k]) for k in range(0, len(esp))]
test_data = list(test_data)

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

# =================== grafico ===================

# eixo = np.array(range(0, 10, 1))
# grafico, v_qnt = corrf(test_data, b, w)
# graf_plot(eixo, grafico)

# =================== numero no arquivo ===================

# print("Quantidade de cada numero no arquivo de teste:")
# for n in range(0,10):
#     print("{}: {}".format(n,v_qnt[n]))

# =================== Proprios Digitos ===================

img_names = glob(os.path.join('Digitos_Proprios', '*.png'))  # Endereco das imagens
img_names.sort() # Ordem Alfabetica

i_test = own_digits(img_names)  # funcao que processa as imagens e retorna uma tupla

# vetor, tamanho, tupla, xv = imagem(i_test, b, w)
# print("Erros computados {}".format(tamanho))
# print("Tupla referente ao erro = {}".format(tupla[2]))
# # print(xv[1].reshape((28, 28)))
# plt.imshow(xv[2].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()

test_results = list()
feedV = list()

for x, y in i_test:
    a = feedforward(x, b, w)
    # print(a)
    feedV += [(a, y)]
    test_results += [(np.argmax(a), y)]
print(test_results)

# acertos = sum(int(x == y) for (x, y) in test_results)
# porc = acertos / len(test_results)
# print("acertos = {}/{}".format(acertos,len(test_results)))
# print("porcentagem de acertos = {:2.2%}".format(porc))


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

# TEST UI

import PySimpleGUI as sg
from PIL import Image, ImageTk

numeros_lista = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7],
             2: [8, 9], 3: [10, 11, 12, 13],
             4: [14, 15], 5: [16, 17, 18],
             6: [19, 20], 7: [21, 22, 23, 24],
             8: [25, 26, 27], 9: [28, 29, 30, 31]}

# numeros = dict(numeros_lista)


# print(numeros_lista[0][1])

# Criar as janelas e estilos(layouts)

def janela_inicio():
    sg.theme('Reddit')
    layout = [
        [sg.Text('Digite o numero:'), sg.Input(key='numero')],
        [sg.Button('Continuar')]
    ]
    return sg.Window('Meu Digito V1.0', layout=layout, finalize=True)


def janela_selecao(img_names, num):
    sg.theme('Reddit')
    if num == 0:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2'),
             sg.Image(img_names[value_list[3]]), sg.Radio('', 1, key='imagem_3')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 1:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2'),
             sg.Image(img_names[value_list[3]]), sg.Radio('', 1, key='imagem_3')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 2:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0')],
            [sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 3:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2'),
             sg.Image(img_names[value_list[3]]), sg.Radio('', 1, key='imagem_3')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 4:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0')],
            [sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 5:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2')],
            # [sg.Text('Sem dados por enquanto!')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 6:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0')],
            [sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            # [sg.Text('Sem dados por enquanto!')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 7:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2'),
             sg.Image(img_names[value_list[3]]), sg.Radio('', 1, key='imagem_3')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 8:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    elif num == 9:
        value_list = numeros_lista[num]
        layout = [
            [sg.Image(img_names[value_list[0]]), sg.Radio('', 1, key='imagem_0'),
             sg.Image(img_names[value_list[1]]), sg.Radio('', 1, key='imagem_1')],
            [sg.Image(img_names[value_list[2]]), sg.Radio('', 1, key='imagem_2'),
             sg.Image(img_names[value_list[3]]), sg.Radio('', 1, key='imagem_3')],
            [sg.Button('Voltar'), sg.Button('Continuar')]
        ]
    else:
        layout = [
            [sg.Text('Por favor, digite um numero entre 0 e 9!')],
            [sg.Button('Voltar')]
        ]

    return sg.Window('Selecao', layout=layout, finalize=True)


def janela_resultado(img_names, num, pos):
    # print(test_results)
    value_list = numeros_lista[num]
    sg.theme('Reddit')
    layout_column = [
        [sg.Image(img_names[value_list[pos]]), sg.Text('-> Numero {}'.format(num))],
        [sg.Text('Rede: {}'.format(test_results[value_list[pos]][0])), sg.Text('Exato: {}'.format(test_results[value_list[pos]][1]))],
        [sg.Text('Vetor feedfoward:\n{}'.format(feedV[value_list[pos]][0]))],
        [sg.Button('Voltar')],
    ]

    layout = [[sg.Column(layout_column, element_justification='center')]]

    return sg.Window('Resultado', layout=layout, resizable=True, grab_anywhere=True, finalize=True)


# Criar as janelas iniciais

janela1, janela2, janela3 = janela_inicio(), None, None

# Criar um loop de leituras de eventos

while True:
    window, event, values = sg.read_all_windows()
    # Quando a janela for fechada
    if window == janela1 and event == sg.WIN_CLOSED:
        exit()
    if window == janela2 and event == sg.WIN_CLOSED:
        exit()
    if window == janela3 and event == sg.WIN_CLOSED:
        exit()
    # Quando queremos ir para a proxima janela
    if window == janela1 and event == 'Continuar':
        if values['numero'] == '0':
            num = 0
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '1':
            num = 1
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '2':
            num = 2
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '3':
            num = 3
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '4':
            num = 4
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '5':
            num = 5
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '6':
            num = 6
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '7':
            num = 7
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '8':
            num = 8
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        elif values['numero'] == '9':
            num = 9
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
        else:
            num = -1
            janela2 = janela_selecao(img_names, num)
            janela1.hide()
    if window == janela2 and event == 'Continuar':
        if event is None:
            exit()
        elif values['imagem_0']:
            pos = 0
            janela3 = janela_resultado(img_names, num, pos)
            janela2.hide()
        elif values['imagem_1']:
            pos = 1
            janela3 = janela_resultado(img_names, num, pos)
            janela2.hide()
        elif values['imagem_2']:
            pos = 2
            janela3 = janela_resultado(img_names, num, pos)
            janela2.hide()
        elif values['imagem_3']:
            pos = 3
            janela3 = janela_resultado(img_names, num, pos)
            janela2.hide()
    # Quando queremos voltar janela anterior
    if window == janela2 and event == 'Voltar':
        janela2.hide()
        janela1.un_hide()
    if window == janela3 and event == 'Voltar':
        janela3.hide()
        janela2.un_hide()
    # Logica de o que deve acontecer ao clicar nos botoes
