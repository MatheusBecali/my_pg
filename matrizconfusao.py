# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   E-mail: matheusbecali@gmail.com
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

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

# Normalization
x_test, y_test = test_set

# Retrive weights
with open(GRAPHIC_DIR + '/SGDM/' + 'pesos_SM_300HL_100e.pkl', 'rb') as handle:
     dataset = pickle.load(handle)

w = (dataset['weights_1'], dataset['weights_2'])
b = (dataset['biases_1'], dataset['biases_2'])

# =================== definicao variaveis ===================

esp = list()
matrix = list()
test_data = list()

# =================== remanejamento dos arquivos para teste ===================

esp += [np.argmax(y_test[i]) for i in range(0, len(y_test))]
matrix += [(x_test[j].reshape(784, 1)) for j in range(0, len(x_test))]
test_data += [(matrix[k], esp[k]) for k in range(0, len(esp))]
test_data = list(test_data)

# =================== funções ===================

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

    for x, y in test_data:
        activation = feedforward(x, b, w)  # Executa o feedforward na rede treinada
        qnt_k[y] += 1  # Quantidade iamgens de cada numeros, onde k = 0, 1, ..., 9
        for i in range(0, 10):
            grafico[y][i] += activation[i]

    for i in range(10):
        for j in range(0, 10):
            grafico[i][j] = (100 * grafico[i][j]) / qnt_k[i]

    return grafico, qnt_k

# # eixo = np.array(range(0, 10, 1))
grafico, v_qnt = corrf(test_data, b, w)
# # grafbar_plot(eixo, grafico)

import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print("Quantidade de cada numero no arquivo de teste:")
for n in range(0,10):
    print("{}: {}".format(n,v_qnt[n]))

test_results = list()

for x, y in test_data:
    a = feedforward(x, b, w)
    test_results += [np.argmax(a)]

data = {'y_Actual':    test_results,
        'y_Predicted': esp
        }



cf_matrix = confusion_matrix(data['y_Actual'], data['y_Predicted'])
df_cm = pd.DataFrame(cf_matrix, index = [i for i in range(10)],
                     columns = [i for i in range(10)])


plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt='g', center=1)
plt.xlabel("Predito") 
plt.ylabel("Esperado") 
# plt.savefig('matriz_de_confusão.pdf')
plt.show()

# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Esperado'], colnames=['Predito'], margins = True)
# sn.heatmap(confusion_matrix, annot=True)
# plt.show()

