import numpy as np

class neuron ():
    def __init__(self,weights=None,act_function=lambda: None,bias=0):
        self.weights=weights
        self.bias=bias
        self.act_function=act_function
        
        self.dWeights=0
        self.dBias=0
    
    def output(self,inputs:np.array):
        #inputs not necessarily always an array. base implementation almost always overriden
        return self.act_function.forward(np.sum(self.weights*inputs)+self.bias)
    
#one neuron per layer
class convNeuron(neuron):
    
    def __init__(self,weights=None,act_function=lambda: None,bias=0,
                    kernelSize=5):
        super().__init__(weights,act_function,bias)
        self.kernelSize=kernelSize #default 5x5 kernel matrices
        #-> inputs should be 5x5 matrices

    def output(self,inputMatrix=None):
        #take a 2d array and run convolution
        
        try:
            np.testing.assert_array_equal(inputMatrix.shape, self.weights.shape)
        except AssertionError as e:
            print(f"weights and input are different sizes: {e}")
        
        return np.sum(inputMatrix*self.weights)
    


