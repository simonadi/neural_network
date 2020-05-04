import numpy as np

from .layer import Layer
from utils import *

class Conv1D(Layer):
    def __init__(self, nfilters, fsize, activation=None):
        self.nfilters = nfilters
        if not isinstance(fsize, tuple):
            sys.exit('Filter size must be passed as a tuple')
        elif len(fsize) > 2:
            sys.exit('Expected filter of 2 dimensions, got {} dimensions')
        self.fsize = fsize
        self.filters = [np.random.normal(size=fsize) for _ in range(nfilters)]
        self.activation_func = activation

    def forward(self, input):
        output = np.zeros(shape=(input.shape[0], input.shape[1], self.nfilters))
        hsize = int((self.fsize[0]-1)/2)
        input = pad(input, self.fsize[0])
        for i in range(output.shape[1]):
            for counter, filter in enumerate(self.filters):
                output[:, i, counter] = np.sum([input[:, hsize + i + j, :]*filter[j+hsize] for j in range(-hsize, hsize+1)], axis=(0,2))#/input.shape[2]
        return output

    def get_input_error(self, y_err):
        pass
