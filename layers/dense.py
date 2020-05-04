import numpy as np

from functions import Unit
from .layer import Layer

class Dense:
    def __init__(self, size, activation=None):
        self.output_size = size[1]
        self.input_size = size[0]
        self.weights = np.random.rand(self.output_size, self.input_size)*0.01
        self.bias = np.zeros(self.output_size)
        if activation == None:
            self.activation = Unit()
        else:
            self.activation = activation

    def forward(self, input):
        output = np.zeros(shape=(input.shape[0], self.output_size))
        for counter, sample in enumerate(input):
            sample = sample.reshape(len(sample), 1)
            output[counter, :] = (self.weights @ sample).reshape(tuple([self.output_size]))
            output[counter, :] += self.bias
        return output, self.activation.compute(output)

    def get_input_error(self, output_error):
        output_error = output_error.reshape(output_error.shape[0], self.output_size)
        error = np.zeros(shape=(output_error.shape[0], self.input_size, 1))
        for index_sample in range(output_error.shape[0]):
            output_error_mat = output_error[index_sample, :][:, np.newaxis]
            weights_mat = self.weights.T
            error[index_sample, :] = weights_mat @ output_error_mat
        return error

    def get_weights_error(self, output_error, input):
        temp = np.swapaxes(input, 1, 2)
        error = output_error @ temp
        return error

    def get_bias_error(self, output_error):
        return output_error

    def get_layer_error(self, out_noac, err):
        a = np.array([error*derinput for error, derinput in zip(err, self.activation.derivative(out_noac))])
        return a

    def update(self, grad_weights, grad_bias, rate):
        self.weights -= rate*np.mean(grad_weights, axis=0)
        self.bias -= rate*(np.mean(grad_bias, axis=0)).reshape(self.output_size)
