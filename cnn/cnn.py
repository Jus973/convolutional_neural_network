from cnn.neuron import convNeuron
import numpy as np

class layer ():
    def __init__(self, numNeurons, neuron):
        self.neuronRow=[]
        for _ in range(numNeurons):
            self.neuronRow.append(neuron())

#implement the convolutional layer with just 1 neuron
#convolution is actually just cross correlation. true convolution has kernel flipped
class convolutional_layer(layer):
    
    def __init__(self, kernelSize=5, stride=1):
        self.kernelSize=kernelSize
        self.stride=stride
        self.cn=convNeuron(kernelSize)
        pass
    
    #implementation of convolution formula assuming it's a square
    def run(self, inputMatrix=np.zeros((28,28))):
        outputSize=len(inputMatrix)-self.kernelSize+1 #TODO alter size depending on stride

        outputMatrix=np.zeros((outputSize, outputSize)) 

        for x in range (0,outputSize,self.stride):
            for y in range (0,outputSize,self.stride):
                
                outputMatrix[x][y] += self.cn.output(inputMatrix[x:x+self.kernelSize,
                                                                y:y+self.kernelSize])    
        
        return outputMatrix
    # All neurons in the same convolutional layer share the same weights


class cnn ():
    def __init__ (self):
        pass
    
    def feed_input(self):
        #input is a tensor of shape (number of inputs x input height x input width x input channels)


        #start the neural network
        pass
    
    

