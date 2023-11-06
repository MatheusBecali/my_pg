# =#= Leitura e treinamento de dados da MNIST =#=#=
#
#   Aluno: Matheus Becali Rocha
#   E-mail: matheusbecali@gmail.com
#
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=


# ========================== Bibliotecas ===========================

import time
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
import pickle
import os
import random
# import matplotlib.cm as cm

np.random.seed(1)
# ============= Importacao dos Dados =============

# NORMAL MNIST
training_set, test_set = load_mnist(normalize=True, one_hot_label=True)

# EXPANDED MNIST
# import gzip
# f = gzip.open("mnist_expanded.pkl.gz", 'rb')
# training_set, test_set = pickle.load(f, encoding="latin1")
# f.close()


def normalize(x, y, test_set):
    """
    Normalize and prepare data for training or testing a neural network.

    Parameters:
    - x (list): A list of input data.
    - y (list): A list of labels or target values.
    - test_set (bool): A boolean flag indicating whether to process the data as a test set (True) or a training set (False).

    Returns:
    If test_set is True:
    - test_data (list): A list of tuples, each containing normalized input data and corresponding labels.
    - n_test (int): The number of data points in the test set.

    If test_set is False:
    - training_data (list): A list of tuples, each containing normalized input data and corresponding labels.
    """
    if test_set:
        # Preparing the test dataset
        esp = list()
        matrix = list()
        test_data = list()
        esp += [np.argmax(y[i]) for i in range(0, len(y))]
        matrix += [(x[j].reshape(784, 1)) for j in range(0, len(x))]
        test_data += [(matrix[k], esp[k]) for k in range(0, len(esp))]

        test_data = list(test_data)
        n_test = len(test_data)

        return test_data, n_test
    
    else:
        # Preparing the training dataset
        esp_t = list()
        matrix_t = list()
        training_data = list()
        esp_t += [y[i].reshape(10, 1) for i in range(0, len(y))]
        matrix_t += [(x[j].reshape(784, 1)) for j in range(0, len(x))]
        training_data += [(matrix_t[k], esp_t[k]) for k in range(0, len(esp_t))]
        
        return training_data


print(len(training_set[0]))
x_train, y_train = training_set
x_test, y_test = test_set

# for i in range(0,100,1):
#     print(f"label ({np.argmax(y_train[i])})")
#     plt.imshow(x_train[i].reshape((28, 28)), cmap=cm.Greys_r)
#     plt.show()

class Network(object):

    def __init__(self, sizes, initialization='random-1'):
        """
        Initialize a neural network with specified layer sizes and weight initialization method.

        Args:
        sizes (list): A list of integers representing the number of neurons in each layer.
        initialization (str): The weight initialization method ('random-1', 'random-2', or 'He').

        Attributes:
        num_layers (int): The total number of layers in the neural network.
        sizes (list): A list of integers representing the number of neurons in each layer.
        weights (list): A list to store the weight matrices for each layer.
        biases (list): A list to store the bias vectors for each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = list()
        self.biases = list()

        if initialization == 'random-1':
            # Random Inicialization
            self.hidden_layer = (np.random.randn(sizes[1], sizes[0]) * 0.01)
            self.output_layer = (np.random.randn(sizes[2], sizes[1]) * 0.01)
            self.weights.append(self.hidden_layer)
            self.weights.append(self.output_layer)
            self.hidden_layer_b = (np.random.randn(sizes[1], 1) * 0.01)
            self.output_layer_b = (np.random.randn(sizes[2], 1) * 0.01)
            self.biases.append(self.hidden_layer_b)
            self.biases.append(self.output_layer_b)
        elif initialization == 'random-2':
            # Random Inicialization 2
            self.hidden_layer = (np.random.rand(sizes[1], sizes[0]) * 2) - (np.ones((sizes[1], sizes[0])))
            self.output_layer = (np.random.rand(sizes[2], sizes[1]) * 2) - (np.ones((sizes[2], sizes[1])))
            self.weights.append(self.hidden_layer)
            self.weights.append(self.output_layer)
            self.hidden_layer_b = (np.random.rand(sizes[1], 1) * 2) - (np.ones((sizes[1], 1)))
            self.output_layer_b = (np.random.rand(sizes[2], 1) * 2) - (np.ones((sizes[2], 1)))
            self.biases.append(self.hidden_layer_b)
            self.biases.append(self.output_layer_b)
        elif initialization == 'He':
            # He Inicialization
            self.hidden_layer = (np.random.randn(sizes[1], sizes[0]) * np.sqrt(2./sizes[0]))
            self.output_layer = (np.random.randn(sizes[2], sizes[1]) * np.sqrt(2./sizes[1]))
            self.weights.append(self.hidden_layer)
            self.weights.append(self.output_layer)
            self.hidden_layer_b = np.zeros((sizes[1], 1))
            self.output_layer_b = np.zeros((sizes[2], 1))
            self.biases.append(self.hidden_layer_b)
            self.biases.append(self.output_layer_b)

        print("weights : {}".format(self.weights[1]))
        

    def feedforward(self, a):
        """
        Perform the feedforward step of the neural network.

        Args:
        a (numpy array): Input features.

        Returns:
        numpy array: The output of the feedforward step.

        This method computes the output of the neural network by propagating the input through the network's layers.
        It applies the sigmoid activation function to the intermediate results and returns the final output.
        """
        b1, b2 = self.biases
        w1, w2 = self.weights
        x1 = np.dot(w1, a) + b1
        a1 = sigmoid(x1)
        x2 = np.dot(w2, a1) + b2
        a2 = sigmoid(x2)
        
        return a2

    def computar_custo(self, a, y):
        """
        Compute the cost (loss) for a predicted output and the corresponding target output.

        Args:
        a (numpy array): Predicted output.
        y (numpy array): Target output.

        Returns:
        float: The cost (loss) value.

        This method computes the cost (loss) between the predicted output and the target output using the squared error loss.
        """
        cost = 0.5 * np.linalg.norm(a - y) ** 2
        cost = np.squeeze(cost)

        return cost


    def total_cost(self, data):
        """
        Calculate the total cost for a dataset.

        Args:
        data (list): List of data pairs (input features and labels).

        Returns:
        float: The total cost (loss) for the given dataset.

        This method calculates the total cost (loss) for a dataset by summing the individual costs for each data point in the dataset.
        The cost is computed using the squared error loss.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.computar_custo(a, y) / len(data)

        return cost

    def cria_mini_batches(self, x_train, y_train, mini_batch_size):
        """
        Create mini-batches from the training data for stochastic gradient descent.

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        mini_batch_size (int): Size of the mini-batches.

        Returns:
        list: A list of mini-batches, where each mini-batch is a list of training data pairs (input features and labels).

        This method takes the input training data and target output training data and divides them into mini-batches
        for use in stochastic gradient descent. The mini-batches are randomly shuffled to ensure randomness in training.
        """
        training_data = normalize(x_train, y_train, test_set=False)
        random.shuffle(training_data)  # Randomize the values
        n = len(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

        return mini_batches

    def SGD(self, x_train, y_train, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using Stochastic Gradient Descent (SGD).

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        epochs (int): The number of training epochs.
        mini_batch_size (int): Size of mini-batches for training.
        eta (float): Learning rate.
        test_data (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of accuracy values on the test data for each epoch.
        list: A list of integers representing the epoch numbers.

        This method trains the neural network using Stochastic Gradient Descent (SGD).
        It updates the network's weights and biases using mini-batches of training data.
        The method returns lists of accuracy values on the test data and epoch numbers to monitor the training progress.
        """
        
        print("Executing SGD...\n")
        test_data = self.test(test_data)
        vet_acertos = list()
        vet_int = list()
        costs = []

        for i in range(epochs):

            mini_batches = self.cria_mini_batches(x_train, y_train, mini_batch_size)
            m = x_train.shape[0]
            # print(len(mini_batches))
            # plt.imshow(x_train[0].reshape((28, 28)), cmap=cm.Greys_r)
            # plt.show()
            cost_total = 0
            for mini_batch in mini_batches:
                
                db = list()
                dw = list()

                for x, y in mini_batch:
                    a3 = self.feedforward(x)
                    cost_total += self.computar_custo(a3, y)

                # cost_total += self.total_cost(mini_batch)
                # a3 = self.feedforward(X)
               
                for x, y in mini_batch:
                    grad = self.backprop(x, y)
                    dw += grad['W']
                    db += grad['b']

                self.weights = [w - eta * nw for w, nw in zip(self.weights, dw)]
                self.biases = [b - eta * nb for b, nb in zip(self.biases, db)]
                
            if test_data:
                vet_acertos.append(self.evaluate(test_data))
                vet_int.append(i)
                print(">epoch {} completed.".format(i))
            else:
                print(">epoch {} completed.".format(i))
            
            # cost_total = self.total_cost(x_train, y_train)
            # costs.append(cost_total)
            # print("Cost on training data: {}".format(costs[-1]))

            cost_avg = cost_total / m
            # cost_avg = np.sum(cost_total)
            if i % 5 == 0 or i == epochs-1:
                print ("Cost after epoch {}: {}".format(i, cost_avg))
            if i % 1 == 0:
                costs.append(cost_avg)
        # vet_acertos = [100 - vet_acertos[i] for i in range(len(vet_acertos))]
        
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 10)')
        plt.title("Learning rate = " + str(eta))
        plt.show()

        return vet_acertos, vet_int

    def SGD_Momentum(self, x_train, y_train, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using Stochastic Gradient Descent (SGD) with Momentum.

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        epochs (int): The number of training epochs.
        mini_batch_size (int): Size of mini-batches for training.
        eta (float): Learning rate.
        test_data (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of accuracy values on the test data for each epoch.
        list: A list of integers representing the epoch numbers.

        This method trains the neural network using Stochastic Gradient Descent (SGD) with Momentum.
        It incorporates a momentum term to accelerate convergence by smoothing out gradient updates.
        The method returns lists of accuracy values on the test data and epoch numbers to monitor the training progress.
        """
        print("Executing SGD with Momentum...\n")
        gamma = 0.9

        vel_biases = list(np.zeros(b.shape) for b in self.biases)
        vel_weights = list(np.zeros(w.shape) for w in self.weights)

        test_data = self.test(test_data)
        vet_acertos = list()
        vet_int = list()

        for i in range(epochs):

            mini_batches = self.cria_mini_batches(x_train, y_train, mini_batch_size)

            for mini_batch in mini_batches:

                db = list()
                dw = list()

                for x, y in mini_batch:
                    grad = self.backprop(x, y)
                    dw += grad['W']
                    db += grad['b']

                vel_weights = [gamma * vw - eta * nw for vw, nw in zip(vel_weights, dw)]
                vel_biases = [gamma * vb - eta * nb for vb, nb in zip(vel_biases, db)]

                self.weights = [w + vw for w, vw in zip(self.weights, vel_weights)]
                self.biases = [b + vb for b, vb in zip(self.biases, vel_biases)]

            if test_data:
                vet_acertos.append(self.evaluate(test_data))
                vet_int.append(i)
                print(">epoch {} completed.".format(i))
            else:
                print(">epoch {} completed.".format(i))

        return vet_acertos, vet_int

    def adagrad(self, x_train, y_train, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using the Adagrad optimization algorithm.

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        epochs (int): The number of training epochs.
        mini_batch_size (int): Size of mini-batches for training.
        eta (float): Learning rate.
        test_data (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of accuracy values on the test data for each epoch.
        list: A list of integers representing the epoch numbers.

        This method trains the neural network using the Adagrad optimization algorithm.
        It adapts the learning rate individually for each parameter by accumulating squared gradients.
        The method returns lists of accuracy values on the test data and epoch numbers to monitor the training progress.
        """

        print("Executing Adagrad...\n")
        eps = 1e-7

        vel_biases = list(np.zeros(b.shape) for b in self.biases)
        vel_weights = list(np.zeros(w.shape) for w in self.weights)

        test_data = self.test(test_data)
        vet_acertos = list()
        vet_int = list()

        for i in range(epochs):

            # vel_biases = list(np.zeros(b.shape) for b in self.biases)
            # vel_weights = list(np.zeros(w.shape) for w in self.weights)

            mini_batches = self.cria_mini_batches(x_train, y_train, mini_batch_size)

            for mini_batch in mini_batches:

                db = list()
                dw = list()

                for x, y in mini_batch:
                    grad = self.backprop(x, y)
                    dw += grad['W']
                    db += grad['b']

                vel_weights = [vw + nw ** 2 for vw, nw in zip(vel_weights, dw)]
                vel_biases = [vb + nb ** 2 for vb, nb in zip(vel_biases, db)]

                self.weights = [w - (eta / np.sqrt(vw + eps)) * nw for w, vw, nw in zip(self.weights, vel_weights, dw)]
                self.biases = [b - (eta / np.sqrt(vb + eps)) * nb for b, vb, nb in zip(self.biases, vel_biases, db)]

            if test_data:
                vet_acertos.append(self.evaluate(test_data))
                vet_int.append(i)
                print(">epoch {} completed.".format(i))
            else:
                print(">epoch {} completed.".format(i))

        return vet_acertos, vet_int

    def adam(self, x_train, y_train, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using the Adam optimization algorithm.

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        epochs (int): The number of training epochs.
        mini_batch_size (int): Size of mini-batches for training.
        eta (float): Learning rate.
        test_data (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of accuracy values on the test data for each epoch.
        list: A list of integers representing the epoch numbers.

        This method trains the neural network using the Adam optimization algorithm.
        It includes both momentum and velocity terms for adaptive learning rate adjustments, along with bias correction.
        The method returns lists of accuracy values on the test data and epoch numbers to monitor the training progress.
        """

        print("Executing Adam...\n")
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8

        mom_biases = list(np.zeros(b.shape) for b in self.biases)
        mom_weights = list(np.zeros(w.shape) for w in self.weights)
        vel_biases = list(np.zeros(b.shape) for b in self.biases)
        vel_weights = list(np.zeros(w.shape) for w in self.weights)

        test_data = self.test(test_data)
        vet_acertos = list()
        vet_int = list()

        costs = []

        for i in range(epochs):

            mini_batches = self.cria_mini_batches(x_train, y_train, mini_batch_size)
            
            cost_total = 0
            m = x_train.shape[0]

            for mini_batch in mini_batches:

                db = list()
                dw = list()

                for x, y in mini_batch:
                    a3 = self.feedforward(x)
                    cost_total += self.computar_custo(a3, y)

                for x, y in mini_batch:
                    grad = self.backprop(x, y)
                    dw += grad['W']
                    db += grad['b']

                mom_weights = [beta_1 * mw + (1 - beta_1) * nw for mw, nw in zip(mom_weights, dw)]
                mom_biases = [beta_1 * mb + (1 - beta_1) * nb for mb, nb in zip(mom_biases, db)]

                vel_weights = [beta_2 * vw + (1 - beta_2) * nw ** 2 for vw, nw in zip(vel_weights, dw)]
                vel_biases = [beta_2 * vb + (1 - beta_2) * nb ** 2 for vb, nb in zip(vel_biases, db)]

                mom_w_chapeu = [mw / (1 - beta_1 ** (i + 1)) for mw in mom_weights]
                mom_b_chapeu = [mb / (1 - beta_1 ** (i + 1)) for mb in mom_biases]

                vel_w_chapeu = [vw / (1 - beta_2 ** (i + 1)) for vw in vel_weights]
                vel_b_chapeu = [vb / (1 - beta_2 ** (i + 1)) for vb in vel_biases]

                self.weights = [w - (eta * mw_c) / (np.sqrt(vw_c) + eps) for w, mw_c, vw_c in
                                zip(self.weights, mom_w_chapeu, vel_w_chapeu)]
                self.biases = [b - (eta * mb_c) / (np.sqrt(vb_c) + eps) for b, mb_c, vb_c in
                               zip(self.biases, mom_b_chapeu, vel_b_chapeu)]

            if test_data:
                vet_acertos.append(self.evaluate(test_data))
                vet_int.append(i)
                print(">epoch {} completed.".format(i))
            else:
                print(">epoch {} completed.".format(i))

            cost_avg = cost_total / m
            # cost_avg = np.sum(cost_total)
            if i % 1 == 0 or i == epochs-1:
                print ("Cost after epoch {}: {}".format(i, cost_avg))
            if i % 1 == 0:
                costs.append(cost_avg)

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 10)')
        plt.title("Learning rate = " + str(eta))
        plt.show()


        return vet_acertos, vet_int

    def rmsprop(self, x_train, y_train, epochs, mini_batch_size, eta, test_data):
        """
        Train the neural network using the RMSprop optimization algorithm.

        Args:
        x_train (numpy array): Input training data.
        y_train (numpy array): Target output training data.
        epochs (int): The number of training epochs.
        mini_batch_size (int): Size of mini-batches for training.
        eta (float): Learning rate.
        test_data (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of accuracy values on the test data for each epoch.
        list: A list of integers representing the epoch numbers.

        This method trains the neural network using the RMSprop optimization algorithm.
        It updates the network's weights and biases in an adaptive manner to improve training efficiency.
        The method returns lists of accuracy values on the test data and epoch numbers for monitoring the training progress.
        """
        print("Executing RMSprop...\n")
        beta = 0.9
        eps = 1e-6

        vel_biases = list(np.zeros(b.shape) for b in self.biases)
        vel_weights = list(np.zeros(w.shape) for w in self.weights)

        test_data = self.test(test_data)
        vet_acertos = list()
        vet_int = list()

        for i in range(epochs):

            mini_batches = self.cria_mini_batches(x_train, y_train, mini_batch_size)

            for mini_batch in mini_batches:

                db = list()
                dw = list()

                for x, y in mini_batch:
                    grad = self.backprop(x, y)
                    dw += grad['W']
                    db += grad['b']

                vel_weights = [beta * vw + (1 - beta) * nw ** 2 for vw, nw in zip(vel_weights, dw)]
                vel_biases = [beta * vb + (1 - beta) * nb ** 2 for vb, nb in zip(vel_biases, db)]

                self.weights = [w - (eta / np.sqrt(vw + eps)) * nw for w, vw, nw in zip(self.weights, vel_weights, dw)]
                self.biases = [b - (eta / np.sqrt(vb + eps)) * nb for b, vb, nb in zip(self.biases, vel_biases, db)]

            if test_data:
                vet_acertos.append(self.evaluate(test_data))
                vet_int.append(i)
                print(">epoch {} completed.".format(i))
            else:
                print(">epoch {} completed.".format(i))

        return vet_acertos, vet_int

    def backprop(self, x, y):
        """
        Perform backpropagation to compute the gradient of the cost function with respect to the network's parameters.

        Args:
        x (numpy array): Input data.
        y (numpy array): Target output.

        Returns:
        dict: A dictionary containing gradients for weights ('W') and biases ('b') for each layer.
        """
        nabla_b = list(np.zeros(b.shape) for b in self.biases)
        nabla_w = list(np.zeros(w.shape) for w in self.weights)

        activation = x
        activations = [x]
        z_error = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_error.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        o_error = self.cost_derivative(y, activations[-1])  # Output layer error
        o_delta = o_error * sigmoid_prime(z_error[-1])  # Applying the derivative of the sigmoid to the error

        nabla_b[-1] = o_delta
        nabla_w[-1] = np.dot(o_delta, activations[-2].T)  # (output --> hidden)

        l = 2
        z2_delta = np.dot(self.weights[-l + 1].T, o_delta) * sigmoid_prime(z_error[-l])
        nabla_w[-l] = np.dot(z2_delta, activations[-l - 1].T)
        nabla_b[-l] = z2_delta

        grad = {'W': nabla_w,
                'b': nabla_b}

        return grad

    def cost_derivative(self, y, output_activations):
        """
        Compute the derivative of the cost function with respect to the output layer activations.

        Args:
        y (numpy array): Target output.
        output_activations (numpy array): Activations of the output layer.

        Returns:
        numpy array: The cost function derivative with respect to the output activations.
        """
        return output_activations - y

    # np.reshape(x, (784, 1))

    # ---------------------------------------------------------------------------------

    def test(self, test_set):
        """
        Prepare and format the test data for evaluation.

        Args:
        test_set (tuple): A tuple containing test data (input features and labels).

        Returns:
        list: A list of tuples, where each tuple contains a reshaped input and its corresponding label.
        """
        x_test, y_test = test_set

        esp = list()
        matrix = list()
        test_data = list()
        esp += [np.argmax(y_test[i]) for i in range(0, len(y_test))]
        matrix += [(x_test[j].reshape(784, 1)) for j in range(0, len(x_test))]
        test_data += [(matrix[k], esp[k]) for k in range(0, len(esp))]

        test_data = list(test_data)
        # n_test = len(test_data)

        return test_data

    def evaluate(self, test_data):
        """
        Evaluate the neural network's performance on the test data.

        Args:
        test_data (list): A list of test data tuples, where each tuple contains a reshaped input and its corresponding label.

        Returns:
        float: The percentage of correct predictions on the test data.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        acertos = sum(int(x == y) for (x, y) in test_results)
        return (acertos  / len(test_results)) * 100

    # ---------------------------------------------------------------------------------

    def predict(self, test_set, dado="test"):
        """
        Print the percentage of correct predictions for the given test set.

        Args:
        test_set (tuple): A tuple containing test data (input features and labels).
        dado (str): A string indicating whether it's a "train" or "test" dataset.

        Returns:
        None

        This method computes and prints the percentage of correct predictions on a given test set.
        It compares the network's predictions to the actual labels in the test data and calculates the accuracy.
        The 'dado' parameter specifies whether it's a training or testing dataset and prints the result accordingly.
        """
        x_test, y_test = test_set

        test_data, n_test = normalize(x_test, y_test, test_set=True)

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        acertos = sum(int(x == y) for (x, y) in test_results)
        porc = acertos / n_test
        if dado == "train":
            print("Percentage of correct predictions on the training set = {:2.2%}".format(porc))
        elif dado == "test":
            print("Percentage of correct predictions on the test set = {:2.2%}".format(porc))


    def saveWeights(self, save_file):
        """
        Save the network's weights and biases to a .pkl file.

        Args:
        save_file (str): The name of the file to be saved as a .pkl file.

        Returns:
        None

        This method saves the weights and biases calculated by the neural network to a .pkl file.
        It stores these values in a dictionary structure within the file, making it easy to access and load the network's weights and biases later.
        """
        pesos = {'weights_1': self.weights[0],
                'biases_1': self.biases[0],
                'weights_2': self.weights[1],
                'biases_2': self.biases[1]}

        # Store the data in a .pkl file
        with open(save_file, 'wb') as handle:
            pickle.dump(pesos, handle, protocol=pickle.HIGHEST_PROTOCOL)



# ================================================================================

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    Args:
    z (float or numpy array): The input value or array to which the sigmoid function is applied.

    Returns:
    float or numpy array: The sigmoid value of the input 'z', mapping it to a range between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid activation function.

    Args:
    z (float or numpy array): The input value or array with respect to which the derivative is calculated.

    Returns:
    float or numpy array: The derivative of the sigmoid function with respect to 'z'.
    """
    return sigmoid(z) * (1 - sigmoid(z))


def saveAxis(vet_acertos, vet_int, save_axis):
    """
    Save axis data to a .pkl file for plotting a graph.

    Args:
    vet_acertos (list): A list of values representing the number of correct predictions.
    vet_int (list): A list of values representing the number of iterations.
    save_axis (str): The name of the file to be saved as a .pkl file.

    Returns:
    None

    This function creates a .pkl file containing the axis data necessary for plotting a graph.
    The 'x' axis represents the number of iterations (vet_int), and the 'y' axis represents the number of correct predictions (vet_acertos).
    """
    axis = {'x': vet_int,
            'y': vet_acertos}

    with open(save_axis, 'wb') as handle:
        pickle.dump(axis, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ================================================================================

def main():
    # Create a neural network with specific layer sizes and initialization
    # initialization='He', initialization='random-1', initialization='random-2'
    net = Network([784, 10, 10], initialization='He')

    start = time.time()

    # Train the network using stochastic gradient descent or others optimizer
    vet_acertos, vet_int = net.SGD(x_train, y_train, 30, 10, 0.001, test_set)
    # vet_acertos, vet_int = net.SGD_Momentum(x_train, y_train, 30, 10, 0.001, test_set
    # vet_acertos, vet_int = net.adam(x_train, y_train, 30, 10, 0.001, test_set)
    # vet_acertos, vet_int = net.adagrad(x_train, y_train, 30, 10, 0.001, test_set)
    # vet_acertos, vet_int = net.rmsprop(x_train, y_train, 30, 10, 0.001, test_set)

    end = time.time()

    # Print the execution time
    print("Tempo de Execucao: {:.2f}".format(end - start))

    # Calculate and display the percentage of correct predictions on training and test sets
    net.predict(training_set, dado="train")
    net.predict(test_set, dado="test")

    # Save axes data
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    save_axis = dataset_dir + '/axis.pkl'
    saveAxis(vet_acertos, vet_int, save_axis)

    # Save the neural network weights
    save_file = dataset_dir + "/pesos.pkl"
    net.saveWeights(save_file)


if __name__ == '__main__':
    main()
