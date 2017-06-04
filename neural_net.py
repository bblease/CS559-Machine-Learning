'''
Ben Blease
4/5/17
A 2 layer neural network for binary classification
Greatly needs optimization
'''

import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt

'''
2 Layer Fully Connected Neural Network
'''
class NN(object):
    def __init__(self, n_hidden):
        self.input = np.zeros(3)
        self.hidden_input = np.zeros((n_hidden, 3))
        self.n_hidden = n_hidden
        self.hidden_weights = np.array([[random.uniform(-1.0, 1.0) for i in range(3)] for k in range(n_hidden)])
        self.hidden = np.zeros(n_hidden)
        self.output_weights = np.array([random.uniform(-1.0, 1.0) for i in range(n_hidden)])
        self.output = 0
        self.hidden_bias = np.array([0 for i in range(n_hidden)])
        self.output_bias = 0

    '''
    Scrub the network of any inputs
    '''
    def clean(self):
        self.input = np.zeros(3)
        self.hidden = np.zeros(self.n_hidden)
        self.output = 0

    '''
    Generate network output
    '''
    def feed_forward(self, inp):
        def relu(a):
            return max(0, np.sum(a))

        self.input = inp
        self.hidden_input = self.input * self.hidden_weights + np.outer(self.hidden_bias, [1, 0, 0])
        #apply ReLu nonlinearity on each hidden node
        self.hidden = np.apply_along_axis(relu, 1, self.hidden_input)
        b = self.hidden * self.output_weights
        self.output = (1/(1 + math.exp(-1 * np.sum(b + self.output_bias))))
        return self.output

    '''
    Return the output error
    '''
    def out_err(self, y):
        err = self.output - y
        return err

    '''
    Return the hidden node error
    '''
    def hidden_err(self, y):
        grad = np.zeros((self.n_hidden, 3))

        def deriv(a):
            b = 1 if (np.sum(a) >= 0) else 0
            return b
        b = np.apply_along_axis(deriv, 1, self.hidden_input)
        grad = (self.output_weights * (self.output - y) * b)
        return grad

    '''
    Adjust weights and biases
    '''
    def backprop(self, g, b, rate, n, lambd):
        self.output_weights = self.output_weights - (rate / n) * g[1] - ((rate / n) * lambd * self.output_weights)
        self.output_bias = self.output_bias - (rate/n) * b[1]
        self.hidden_weights = self.hidden_weights - (rate / n) * g[0] - ((rate / n) * lambd * self.hidden_weights)
        self.output_bias = self.output_bias - (rate/n) * b[0]

'''
Make the dataset
'''
def make_set(a, b, n):
    return [[random.uniform(a, b) for i in range(3)] for j in range(n)]

'''
Make the test dataset
'''
def make_y(arr):
    out = []
    for i in arr:
        sum = 0
        for j in i:
            sum += math.floor(j)
        out += [sum % 2]
    return out

'''
Run the network on training and test sets
'''
def runNN(X, Y, hidden, rate, it, lambd, tx, ty):
    new_net = NN(hidden)

    #initialize plotting functions
    plt.axis = [0, 1, 0, 1]
    fig = plt.figure()
    a1 = fig.add_subplot(2, 1, 1)
    a2 = fig.add_subplot(2, 1, 2)
    plt.ion()

    for i in range(it):
        g = [0, 0]
        b = [0, 0]
        c_ent = 0.0
        total = 0.0
        c_ent_test = 0.0
        total_test = 0.0

        for j in range(len(X)):
            new = new_net.feed_forward(X[j])
            g[0] = g[0] + np.outer(new_net.hidden_err(Y[j]), new_net.input)
            g[1] = g[1] + np.multiply(new_net.out_err(Y[j]), new_net.hidden)

            b[1] = b[1] + new_net.out_err(Y[j])
            b[0] = b[0] + new_net.hidden_err(Y[j])

            answer = 1 if (new > 0.5) else 0
            if answer == Y[j]: total += 1
            c_ent += (Y[j] * math.log(new) + (1 - Y[j]) * math.log(1 - new) )
            new_net.clean()

        new_net.backprop(g, b, rate, len(X), lambd)

        for j in range(len(tx)):
            new = new_net.feed_forward(tx[j])
            answer = 1 if (new > 0.5) else 0
            if answer == ty[j]: total_test += 1
            c_ent_test += (ty[j] * math.log(new) + (1 - ty[j]) * math.log(1 - new) )

        #plot the values
        a1.axis([0, i + 20, 0, 1])
        a1.scatter(i, -c_ent_test/len(tx), color='red', s=5, marker=',')
        a1.scatter(i, -c_ent/len(X), color='blue', s=5, marker=',')
        a2.axis([0, i + 20, 0, 1])
        a2.scatter(i, 1 - total_test/len(tx), color='red', s=5, marker=',')
        a2.scatter(i, 1 - total/len(X), color='blue', s=5, marker=',')
        plt.pause(0.00001)

        if i % 20 == 0:
            print("Guessed correctly : " + str(total/len(X)) + ", Cross Entropy : " + str(-c_ent/len(X)) + ", Test Guessed Correctly : " + str(total_test/len(tx)) + ", Test Cross Entropy : " + str(-c_ent_test/len(tx)) + ", @ Iteration " + str(i))

def main(argc, argv):
    if argc != 8:
        print("Usage : <n hidden> <learning rate> <iterations> <regularizer> <set bound> <training set size> <test set size>")

    else:
        x = make_set(0, int(argv[5]), int(argv[6]))
        y = make_y(x)
        tx = make_set(0, int(argv[5]), int(argv[7]))
        ty = make_y(tx)
        runNN(x, y, int(argv[1]), float(argv[2]), int(argv[3]), float(argv[4]), tx, ty)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
