import numpy as np
from functions import Unit

def kernel_middle(size):
    k = np.zeros(size)
    k[int((size[0]-1)/2), :] = np.ones(shape=(size[1]))
    return k

def pad(vects, fsize):
    hsize = int((fsize-1)/2)
    output = np.zeros(shape=(vects.shape[0], vects.shape[1] + 2*hsize, vects.shape[2]))
    output[:, hsize:-hsize, :] = vects
    return output

class Conv1D:
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
        # print(input[0])
        for i in range(output.shape[1]):
            for counter, filter in enumerate(self.filters):
                output[:, i, counter] = np.sum([input[:, hsize + i + j, :]*filter[j+hsize] for j in range(-hsize, hsize+1)], axis=(0,2))#/input.shape[2]
        return output
        # return self.activation_func.f(output)

    def backward(self, y_err):
        pass

class MaxPooling1D:
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

    def get_weights_error(self):
        return 0.

    def update(self, grad, rate):
        pass

class Dense:
    def __init__(self, size, activation=None):
        self.output_size = size[1]
        self.input_size = size[0]
        # self.weights = np.random.normal(size=(self.output_size, self.input_size))*10
        self.weights = np.random.rand(self.output_size, self.input_size)*0.01
        self.bias = np.zeros(self.output_size)
        if activation == None:
            self.activation = Unit()
        else:
            self.activation = activation

    def forward(self, input):
        # print('mean weight : ' ,np.mean(self.weights, axis=(0,1)))
        output = np.zeros(shape=(input.shape[0], self.output_size))
        for counter, sample in enumerate(input):
            sample = sample.reshape(len(sample), 1)
            output[counter, :] = (self.weights @ sample).reshape(tuple([self.output_size]))
            output[counter, :] += self.bias
        # print(output.shape)
        return output, self.activation.compute(output)

    def get_input_error(self, output_error):
        # print('INPUT ERROR COMPUTING : OUTPUT ERROR SHAPE : ', output_error.shape)
        output_error = output_error.reshape(output_error.shape[0], self.output_size)
        error = np.zeros(shape=(output_error.shape[0], self.input_size, 1))
        for index_sample in range(output_error.shape[0]):
            for index_input in range(self.input_size):
                error[index_sample, index_input] = sum([self.weights[index_output, index_input]*output_error[index_sample, index_output] for index_output in range(self.output_size)])
        return error

    def get_weights_error(self, output_error, input):
        # print('WEIGHTS ERROR COMPUTING : OUTPUT ERROR SHAPE : ', output_error.shape)
        # print('ouput size : ', self.output_size)
        error = np.zeros(shape=tuple([input.shape[0]]) + self.weights.shape)
        # print(output_error)
        for index_output in range(self.output_size):
            for index_input in range(self.input_size):
                for index_sample in range(input.shape[0]):
                    error[index_sample, index_output, index_input] = input[index_sample, index_input, 0]*output_error[index_sample, index_output, 0]
        # print(error)
        return error

    def get_bias_error(self, output_error):
        # print('error shape bias : ', output_error.shape)
        return output_error

    def get_layer_error(self, out_noac, err):
        # print('LAYERS ERROR COMPUTING : OUTPUT ERROR SHAPE : ', err.shape)
        # print('LAYERS ERROR COMPUTING : OUTNOAC SHAPE : ', out_noac.shape)
        b = self.activation.derivative(out_noac)
        # print('B : ', b)
        # print('B SHAPE : ', b.shape)
        a = np.array([error*derinput for error, derinput in zip(err, self.activation.derivative(out_noac))])
        # print('AAAAAAAA : ', a.shape)
        return a

    def update(self, grad_weights, grad_bias, rate):
        self.weights -= rate*np.mean(grad_weights, axis=0)
        # print('grad bias : ', grad_bias.shape)
        # print('bias shape : ', self.bias.shape)
        self.bias -= rate*(np.mean(grad_bias, axis=0)).reshape(self.output_size)

if __name__ == '__main__':
    dense = Dense((100, 10))
    conv = Conv1D(100, (5,3), kernel_middle)
    gmax = MaxPooling1D((100, 5))