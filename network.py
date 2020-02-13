import numpy as np
import sys
from pde_testing.tools_pde import *
import math
from functions import BinaryCrossEntropy, Sigmoid
from multiprocessing import Pool
from functools import reduce

from layers import Dense, Conv1D, MaxPooling1D

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

class Network:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def train(self, inputs, labels, epochs=10, batch_size=1):
        nbatch = math.ceil(inputs.shape[0]/batch_size)
        batches_inputs = [inputs[i*batch_size:(i+1)*batch_size] for i in range(nbatch-1)] + [inputs[(nbatch-1)*batch_size:]]
        batches_labels = [labels[i*batch_size:(i+1)*batch_size] for i in range(nbatch-1)] + [labels[(nbatch-1)*batch_size:]]
        for epoch in range(1, epochs+1):
            print('Epoch {}/{}'.format(epoch, epochs))
            for batch_inputs, batch_labels in zip(batches_inputs, batches_labels):
                forward_data = batch_inputs
                outputs_noac = [batch_inputs]
                outputs_ac = []
                for l in self.layers:
                    out_noac, forward_data = l.forward(forward_data)
                    outputs_ac.append(forward_data)
                    outputs_noac.append(out_noac)
                forward_data = forward_data.reshape(tuple([forward_data.shape[0]]))
                print('Accuracy : ', accuracy(forward_data, batch_labels))
                loss_final = self.loss.derivative(forward_data, batch_labels)
                print('Loss : {}'.format(abs(np.mean(self.loss.compute(forward_data, batch_labels)))))
                backward_err = loss_final
                gradients_weights = []
                gradients_bias = []
                i = 1
                out_noac = outputs_noac[-i]
                out_noac = out_noac.reshape(out_noac.shape[0], out_noac.shape[1], 1)
                for l in reversed(self.layers):
                    error = l.get_layer_error(out_noac, backward_err)
                    i += 1
                    out_noac = outputs_noac[-i]
                    out_noac = out_noac.reshape(out_noac.shape[0], out_noac.shape[1], 1)
                    gradients_weights.append(l.get_weights_error(error, out_noac))
                    gradients_bias.append(l.get_bias_error(error))
                    backward_err = l.get_input_error(error)
                for l, grad_w, grad_b in zip(self.layers, reversed(gradients_weights), reversed(gradients_bias)):
                    l.update(grad_w, grad_b, 0.1)

if __name__ == '__main__':
    solutions, labels = get_data(np.linspace(1, 5, 100), np.linspace(0.5, 1.5, 10))
    solutions, labels = unison_shuffled_copies(solutions, labels)

    sigmoid = Sigmoid()

    bce = BinaryCrossEntropy()

    layers = [Dense((1001, 100)),
              Dense((100, 50)),
              Dense((50, 1), activation=sigmoid)]

    network = Network(layers, bce)
    network.train(solutions, labels, epochs=25, batch_size=50)
