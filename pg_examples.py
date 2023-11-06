# =#=#=#=#=#=#=#=#= PG Examples =#=#=#=#=#=#=#=#
#
#   Aluno: Matheus Becali Rocha
#   Matricula: 2017101659
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

import numpy as np

def compute_cost(A2, Y):

    m = Y.shape[1] # number of examples
    
    cost = (1/m) * 0.5*np.linalg.norm(A2-Y)**2

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
    
    return cost

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# b = (np.array([[-0.3],[0.82],[0.03]]), np.array([[-0.8],[0.1]]))
# w = (np.array([[0.1,-0.5, 0.9],[0.4, 0.1,0.5]]), np.array([[0.8, 0.1],[-0.12, 0.1],[0.3, 0.1]]))

##### Sem w.T

# b = (np.array([[-0.3],[0.82],[0.03]]), np.array([[-0.8]]))
# w = (np.array([[0.1,0.4],[-0.5, 0.1],[0.9, 0.5]]), np.array([[0.8, -0.12,0.3]]))

parametros = {
    "w1": np.array([[0.1,0.4],[-0.5, 0.1],[0.9, 0.5]]),
    "w2": np.array([[0.8, -0.12,0.3]]),
    "b1": np.array([[-0.3],[0.82],[0.03]]),
    "b2": np.array([[-0.8]])
}

def feedforward(a0, parametros):
        
        x = np.dot(parametros["w1"], a0) + parametros["b1"]
        a1 = sigmoid(x)
        x = np.dot(parametros["w2"], a1) + parametros["b2"]
        a2  = sigmoid(x)
        return a2

x = np.array([[0.5],[0.2]])

# print("Feed-Forward: {}".format(feedforward(x,w,b)))

# y = np.array([[0.9],[0.1]])
y = np.array([[0.9]])

def backprop(x, y, parametros):

        biases = (parametros["b1"], parametros["b2"])
        weights = (parametros["w1"], parametros["w2"])
        
        nabla_b = list(np.zeros(b.shape) for b in biases)
        nabla_w = list(np.zeros(w.shape) for w in weights)

        # feedforward
        activation = x
        activations = [x]
        z_error = []

        for b, w in zip(biases, weights):
            z = np.dot(w, activation) + b
            z_error.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backpropagation
        o_error = (activations[-1] - y)  # erro na output
        o_delta = o_error * sigmoid_prime(z_error[-1])  # aplicando a derivada do sigmoid no erro
        print(f"Erro da Output: {o_delta}")
        nabla_b[-1] = o_delta
        nabla_w[-1] = np.dot(o_delta, activations[-2].T)  # (output --> hidden)
        z2_delta = np.dot(weights[-1].T, o_delta) * sigmoid_prime(z_error[-2])
        nabla_w[-2] = np.dot(z2_delta, activations[-3].T)
        nabla_b[-2] = z2_delta
        
        grad = {'W': nabla_w,
                'b': nabla_b}

        return grad

# pesos = backprop(x,y,w,b)

# print(f"w = {w}")
# print(f"b = {pesos['b'][1]}")

def treinamento(x,y,parametros):

    for i in range(0, 100):

        A2 = feedforward(x, parametros)
        cost = compute_cost(A2, y)

        grad = backprop(x, y, parametros)

        for l in range(0,2):
            parametros["w" + str(l+1)] = parametros["w" + str(l+1)] - 0.5 * grad['W'][l]
            # w2 = w2 - 0.5 * grad['W'][1]
            parametros["b" + str(l+1)] = parametros["b" + str(l+1)] - 0.5 * grad['b'][l]
            # b2 = b2 - 0.5 * grad['b'][1]
        
        if i % 1 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parametros

parametros = treinamento(x, y, parametros)

print("Feed-Forward: {}".format(feedforward(x, parametros)))