# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   Matricula: 2017101659
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

# =================== abertura e armazenamento de dados ===================

import pickle

n_hl = 50 # muda o range(0, x) caso altere as hidden layers

def weights():
    w = list()
    w1 = list()
    w2 = list()
    aux2_w1 = list()
    aux2_w2 = list()

    f_w1 = open("weights_1.txt", "r")
    aux_w1 = f_w1.readlines()
    aux2_w1 += [aux_w1[i].split(' ') for i in range(0, n_hl)]
    for i in range(0, n_hl):
        w1_l = list()
        w1_l += [float(aux2_w1[i][j]) for j in range(0, 784)]
        w1.append(w1_l)
    f_w1.close()

    f_w2 = open("weights_2.txt", "r")
    aux_w2 = f_w2.readlines()
    aux2_w2 += [aux_w2[i].split(' ') for i in range(0, 10)]
    for i in range(0, 10):
        w2_l = list()
        w2_l += [float(aux2_w2[i][j]) for j in range(0, n_hl)]
        w2.append(w2_l)
    f_w2.close()

    w.append(w1)
    w.append(w2)

    return w


def biases():
    b = list()
    b1 = list()
    b2 = list()
    aux2_b1 = list()
    aux2_b2 = list()

    f_b1 = open("biases_1.txt", "r")
    aux_b1 = f_b1.readlines()
    aux2_b1 += [aux_b1[i].split(' ') for i in range(0, n_hl)]
    for i in range(0, n_hl):
        b1_l = list()
        b1_l += [float(aux2_b1[i][0])]
        b1.append(b1_l)
    f_b1.close()

    f_b2 = open("biases_2.txt", "r")
    aux_b2 = f_b2.readlines()
    aux2_b2 += [aux_b2[i].split(' ') for i in range(0, 10)]
    for i in range(0, 10):
        b2_l = list()
        b2_l.append(float(aux2_b2[i][0]))
        b2.append(b2_l)
    f_b2.close()

    b.append(b1)
    b.append(b2)

    return b