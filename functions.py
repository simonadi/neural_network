import numpy as np
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def compute(self, output, labels):
        loss = -(1/len(output))*sum([labels[i]*np.log(output[i]) + (1 - labels[i])*np.log(1 - output[i]) for i in range(len(output))])
        return loss

    def derivative(self, output, labels):
        d_losses = np.zeros(shape=(len(output), 1, 1))
        for i in range(len(output)):
            if output[i] == labels[i]:
                d_losses[i] = 0
            elif output[i] == 0. or output[i]==1.:
                d_losses[i] = float('inf')
            else:
                d_losses[i] = (labels[i] - output[i])/(output[i]*(output[i]-1))
        return(d_losses)

class Sigmoid:
    def __init__(self):
        pass
    def compute(self, x):
        output = np.zeros(shape=x.shape)
        for counter, value in enumerate(x):
            output[counter] = 1/(1+np.exp(-value))
        return output
    def derivative(self, x):
        result = np.zeros(shape=x.shape)
        for index_sample in range(result.shape[0]):
            for index_neuron in range(result.shape[1]):
                result[index_sample, index_neuron, 0] = sigmoid(x[index_sample, index_neuron, 0])*(1 - sigmoid(x[index_sample, index_neuron, 0]))
        return result

class Unit:
    def __init__(self):
        pass
    def compute(self, x):
        return x
    def derivative(self, x):
        return np.ones(shape=x.shape)
