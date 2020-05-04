import numpy as np

from .layer import Layer

class MaxPooling1D(Layer):
    def __init__(self, size):
        self.indexes = None
        self.input_size = size
        self.output_size = (size[1], 1)

    def forward(self, input):
        output = np.zeros(shape=(input.shape[0], input.shape[2]))
        indexes = np.zeros(shape=(input.shape[0], input.shape[2]))
        for index_sample in range(input.shape[0]):
            for index_feature in range(input.shape[2]):
                max_val = float('-inf')
                index_max = None
                for counter, item in enumerate(input[index_sample, :, index_feature]):
                    if item > max_val:
                        max_val = item
                        index_max = counter
                output[index_sample, index_feature] = max_val
                indexes[index_sample, index_feature] = index_max
        return output, indexes

    def get_input_error(self, output_error, indexes):
        error = np.zeros(shape=self.input_size)
        for index_feature in range(self.input_size[1]):
            error[indexes[index_feature], index_feature] = output_error[index_feature]
        return error
