from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# A simple sigmoid function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# A sigmoid's derivative
def sigmoid_derivative(x):
    return x * (1.0 - x)


# To represent our neural network
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    #X = np.array([[0, 0, 1],
    #              [0, 1, 1],
    #              [1, 0, 1],
    #              [1, 1, 1]])
    #y = np.array([[0], [1], [1], [0]])
    dt = pd.read_csv(r"data_ffnn_3classes.txt", sep='\s+')

    X1 = dt.iloc[:,0].to_numpy()
    X2 = dt.iloc[:,1].to_numpy()
    Y = dt.iloc[:,2].to_numpy()

    X = np.hstack(np.array(X1[:]))
    X = np.hstack(np.array(X2[:]))
    X = np.hstack(np.ones(len(X1)))
    X = X/np.max(X)
    Y= Y/np.max(Y)
    nn = NeuralNetwork(X, Y)

    for i in range(200):
        nn.feedforward()
        nn.backprop()

    print(nn.output)

plt.plot(y, 'go')
plt.plot(nn.output, 'ro')
plt.title('NN accuracy versus expected\nGreen = Actual, Red = NN Prediction')
plt.ylabel('Y Value')
plt.xlabel('Corresponding X array')
plt.show()