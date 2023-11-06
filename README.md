# Neural Network Class (Documentation)

## Introduction

The `Network` class represents a neural network that can be used for training, testing, and prediction tasks. This neural network can be configured with different initialization methods and optimization algorithms for training. The class includes methods for training the network, making predictions, and saving the trained model's weights and biases.

## Initialization

### Constructor: `__init__(self, sizes, initialization='random-1')`

The constructor initializes a neural network with specified layer sizes and an optional initialization method.

- `sizes` (list): A list specifying the number of neurons in each layer, where the first element is the input layer size, the last element is the output layer size, and any intermediate elements are the sizes of hidden layers.

- `initialization` (str): The initialization method for weights and biases. It can be one of the following:
    - `'random-1'`: Random initialization with small values.
    - `'random-2'`: Random initialization with larger values.
    - `'He'`: He initialization with ReLU activation.

## Feedforward

### Method: `feedforward(self, a)`

This method performs the feedforward step of the neural network.

- `a` (numpy array): Input features.

It returns the output of the feedforward step.

## Cost Functions

### Method: `computar_custo(self, a, y)`

This method computes the cost (loss) for a predicted output and the corresponding target output using the squared error loss.

- `a` (numpy array): Predicted output.
- `y` (numpy array): Target output.

It returns the cost (loss) value.

### Method: `total_cost(self, data)`

This method calculates the total cost for a dataset by summing the individual costs for each data point in the dataset. The cost is computed using the squared error loss.

- `data` (list): List of data pairs (input features and labels).

It returns the total cost (loss) for the given dataset.

## Mini-Batches

### Method: `cria_mini_batches(self, x_train, y_train, mini_batch_size)`

This method creates mini-batches from the training data for stochastic gradient descent.

- `x_train` (numpy array): Input training data.
- `y_train` (numpy array): Target output training data.
- `mini_batch_size` (int): Size of the mini-batches.

It returns a list of mini-batches, where each mini-batch is a list of training data pairs (input features and labels).

## Training

### Method: `SGD(self, x_train, y_train, epochs, mini_batch_size, eta, test_data)`

This method trains the neural network using Stochastic Gradient Descent (SGD) with different optimization algorithms.

- `x_train` (numpy array): Input training data.
- `y_train` (numpy array): Target output training data.
- `epochs` (int): The number of training epochs.
- `mini_batch_size` (int): Size of mini-batches for training.
- `eta` (float): Learning rate.
- `test_data` (tuple): A tuple containing test data (input features and labels).

It returns lists of accuracy values on the test data and epoch numbers to monitor the training progress.

### Method: `SGD_Momentum(self, x_train, y_train, epochs, mini_batch_size, eta, test_data)`

This method trains the neural network using Stochastic Gradient Descent (SGD) with Momentum.

It returns lists of accuracy values on the test data and epoch numbers.

### Method: `adagrad(self, x_train, y_train, epochs, mini_batch_size, eta, test_data)`

This method trains the neural network using the Adagrad optimization algorithm.

It returns lists of accuracy values on the test data and epoch numbers.

### Method: `adam(self, x_train, y_train, epochs, mini_batch_size, eta, test_data)`

This method trains the neural network using the Adam optimization algorithm.

It returns lists of accuracy values on the test data and epoch numbers.

### Method: `rmsprop(self, x_train, y_train, epochs, mini_batch_size, eta, test_data)`

This method trains the neural network using the RMSprop optimization algorithm.

It returns lists of accuracy values on the test data and epoch numbers.

## Backpropagation

### Method: `backprop(self, x, y)`

This method performs backpropagation to compute the gradient of the cost function with respect to the network's parameters.

- `x` (numpy array): Input data.
- `y` (numpy array): Target output.

It returns a dictionary containing gradients for weights ('W') and biases ('b') for each layer.

## Cost Derivative

### Method: `cost_derivative(self, y, output_activations)`

This method computes the derivative of the cost function with respect to the output layer activations.

- `y` (numpy array): Target output.
- `output_activations` (numpy array): Activations of the output layer.

It returns the cost function derivative with respect to the output activations.

## Test Data Preparation

### Method: `test(self, test_set)`

This method prepares and formats the test data for evaluation.

- `test_set` (tuple): A tuple containing test data (input features and labels).

It returns a list of tuples, where each tuple contains a reshaped input and its corresponding label.

## Evaluation

### Method: `evaluate(self, test_data)`

This method evaluates the neural network's performance on the test data.

- `test_data` (list): A list of test data tuples, where each tuple contains a reshaped input and its corresponding label.

It returns the percentage of correct predictions on the test data.

## Prediction

### Method: `predict(self, test_set, dado="test")`

This method prints the percentage of correct predictions for the given test set.

- `test_set` (tuple): A tuple containing test data (input features and labels).
- `dado` (str): A string indicating whether it's a "train" or "test" dataset.

## Saving Weights

### Method: `saveWeights(self, save_file)`

This method saves the network's weights and biases to a .pkl file.

- `save_file` (str): The name of the file to be saved as a .pkl file.

The method stores these values in a dictionary structure within the file, making it easy to access and load the network's weights and biases later.
