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
    
    def __init__(self, kernelSize=3, stride=1,padding=0):
        self.kernelSize=kernelSize
        self.stride=stride
        self.cn=convNeuron(kernelSize)
        self.padding=padding
    
    #implementation of convolution formula assuming it's a square
    def run(self, inputMatrix=np.zeros((28,28))):
        
        paddedMatrix = np.pad(
            inputMatrix,
            ((self.padding, self.padding),
            (self.padding, self.padding)),
            mode="constant"
        )

        outputSize=len(paddedMatrix)-self.kernelSize+1#TODO alter size depending on stride
        outputMatrix=np.zeros((outputSize, outputSize)) 
        
        for x in range (0,outputSize):
            for y in range (0,outputSize):
                iX=x*self.stride
                iY=y*self.stride

                outputMatrix[x][y] = self.cn.output(paddedMatrix[iX:iX+self.kernelSize,
                                                                iY:iY+self.kernelSize])    
        
        return outputMatrix
    # All neurons in the same convolutional layer share the same weights

class reLU (layer):
    def __init__(self):
        pass
    
    def run(self, inputMatrix=np.zeros((28,28))):
        rectifiedUnit=lambda x:max(0,x)
        vectorizedUnit=np.vectorize(rectifiedUnit)
        return vectorizedUnit(inputMatrix)


class max_pooling_layer(layer):
    def __init__(self, kernelSize=2, stride=2):
        self.kernelSize=kernelSize
        self.stride=stride
    
    def run(self, inputMatrix=np.zeros((28,28))):
        if len(inputMatrix) % 2 != 0:
            raise ValueError("2x2 maxpool requires %2==0")
    
        outputSize=len(inputMatrix)//self.kernelSize
        outputMatrix=np.zeros((outputSize, outputSize)) 

        for x in range (0,outputSize):
            for y in range (0,outputSize):
                
                iX=x*self.stride
                iY=y*self.stride

                outputMatrix[x][y] += np.max(inputMatrix[iX:iX+self.kernelSize,
                                                    iY:iY+self.kernelSize])    
        
        return outputMatrix


class convolutional_block ():
    def __init__(self, conv=convolutional_layer(3,1,1), 
                    activationFunction=reLU(), 
                    pool=max_pooling_layer(2, 2)):
        self.conv=conv
        self.activationFunction=activationFunction
        self.pool=pool

    def run (self, inputMatrix):
        p1=self.conv.run(inputMatrix)
        p2=self.activationFunction.run(p1)
        p3=self.pool.run(p2)
        
        return p3



class cnn ():
    def __init__ (self):
        pass
    
    def feed_input(self):
        #input is a tensor of shape (number of inputs x input height x input width x input channels)


        #start the neural network
        pass
    
    

