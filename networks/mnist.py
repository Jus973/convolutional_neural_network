#test my architecture structure + accuracy on the MNIST dataset
#leNET structure

from cnn.cnn import cnn
from cnn.cnn import convolutional_block
from cnn.cnn import convolutional_layer
from cnn.cnn import layer
import numpy as np


class tanh (layer):

    def forward(self, inputMatrix=None):
        self.inputMatrix=inputMatrix
        self.outputMatrix=np.tanh(inputMatrix)
        return self.outputMatrix

    def backward(self, outputMatrix):
        return outputMatrix * (1 - np.tanh(self.inputMatrix)**2)


'''
LeNET structure

32x32 input
normalization need to implement in here

c1 layer:
6 layers. 5x5 kernel size
0 padding
1 stride
-> 28x28x6


'''