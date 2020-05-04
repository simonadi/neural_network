from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def get_input_error(self, output_error):
        pass

    @abstractmethod
    def get_weights_error(self, output_error, input):
        pass

    @abstractmethod
    def get_bias_error(self, output_error):
        pass

    @abstractmethod
    def get_layer_error(self, out_noac, err):
        pass

    @abstractmethod
    def update(self, grad_weights, grad_bias, rate):
        pass
