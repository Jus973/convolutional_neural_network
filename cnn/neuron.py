import numpy as np

class neuron ():
    def __init__(self,weights:np.array=None,act_function=lambda: None,bias=0):
        self.weights=weights
        self.bias=0
        self.act=act_function
    
    def output(self,inputs:np.array):
        return self.act_function(np.sum(self.weights*inputs)+self.bias)
    
    
class convNeuron(neuron):
    weights=None #shared weights within a convolutional layer

    def __init__(self, weights):
        super.__init__()
        convNeuron.weights=weights

    
        
    