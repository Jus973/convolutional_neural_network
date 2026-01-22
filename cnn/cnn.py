from cnn.neuron import convNeuron
from cnn.neuron import neuron
import numpy as np

class layer ():

    def __init__(self):
        self.inputMatrix=None
        self.outputMatrix=None
    
    def getInput(self):
        return self.inputMatrix

    def getOutput(self):
        return self.outputMatrix
    
    def forward(self):
        pass

    def backward(self):
        pass


#implement the convolutional layer with just 1 neuron
#convolution is actually just cross correlation. true convolution has kernel flipped
class convolutional_layer(layer):
    
    def __init__(self, kernelSize=3, stride=1,padding=0):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
        self.cn=convNeuron(kernelSize)
        self.padding=padding
    
    #implementation of convolution formula assuming it's a square
    def forward(self, inputMatrix=np.zeros((28,28))):
        
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
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix
        return outputMatrix
    # All neurons in the same convolutional layer share the same weights


class reLU (layer):

    def forward (self, inputMatrix=np.zeros((28,28))):
        rectifiedUnit=lambda x:max(0,x)
        vectorizedUnit=np.vectorize(rectifiedUnit)

        self.inputMatrix=inputMatrix
        self.outputMatrix=vectorizedUnit(inputMatrix)
        return self.outputMatrix

class softmax (layer):

    def forward (self, inputMatrix=[]):
        summation=sum(np.exp(inputMatrix))
        outputMatrix=[np.exp(x)/summation for x in inputMatrix]

        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix

        return outputMatrix

class max_pooling_layer(layer):
    def __init__(self, kernelSize=2, stride=2):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
    
    def forward(self, inputMatrix=np.zeros((28,28))):
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
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix

        return outputMatrix

class fully_connected_layer(layer):

    def __init__(self, numNeurons, neuronType, act_function):
        super().__init__()
        self.neuronRow=[]
        for _ in range(numNeurons):
            self.neuronRow.append(neuronType(act_function=act_function)) #TODO implement weights

    def forward(self, inputMatrix): #input matrix is flattened vector
        outputMatrix=[]
        for i in range(len(self.neuronRow)):
            #represent vector horizontally
            outputMatrix.append(self.neuronRow[i].output(inputMatrix))
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix
        return outputMatrix


#clean way of combining layers to be blocks while still storing outputs
class block ():
    def __init__(self):
        self.layerOutputs=[]
        self.currentOutput=None

    def performLayer(self, forwardFunction):
        self.layerOutputs.append(forwardFunction(self.currentOutput))


class convolutional_block (block):
    def __init__(self, conv=convolutional_layer(3,1,1), 
                    activationFunction=reLU, 
                    pool=max_pooling_layer(2, 2)):
        self.conv=conv
        self.activationFunction=activationFunction
        self.pool=pool

    #convolutional_layer.convNeuron.weights are my weights

    def forward (self, inputMatrix):
        self.currentOutput=inputMatrix

        self.performLayer(self.conv.forward)
        self.performLayer(self.activationFunction.forward)
        self.performLayer(self.pool.forward)

        return self.currentOutput

class classifier (block):
    def __init__(self, full1=fully_connected_layer(6,neuron,reLU),
                        full2=fully_connected_layer(3,neuron,reLU)):
        self.full1=full1
        self.full2=full2

        #fully_connected_layer.neuronRow[x] are my weights for x neurons
    
    def forward (self, inputMatrix):

        self.currentOutput=inputMatrix

        self.performLayer(np.ndarray.flatten)
        self.performLayer(self.full1.forward)
        self.performLayer(self.full2.forward)
        self.performLayer(softmax)

        return self.currentOutput
    

class cnn ():
    def __init__ (self):
        self.b1=convolutional_block(conv=convolutional_layer(3,1,1), 
                        activationFunction=reLU, 
                        pool=max_pooling_layer(2, 2))
        self.b2=convolutional_block(conv=convolutional_layer(3,1,1), 
                        activationFunction=reLU, 
                        pool=max_pooling_layer(2, 2))
        self.b3=convolutional_block(conv=convolutional_layer(3,1,1), 
                        activationFunction=reLU, 
                        pool=max_pooling_layer(2, 2))
        self.c1=classifier(full1=fully_connected_layer(128,neuron,reLU),
                        full2=fully_connected_layer(10,neuron,reLU))
        
        self.blocks=[self.b1, self.b2, self.b3, self.c1]
        self.layerOutputs=[] 

        self.weightMappings=[self.b1.conv.cn.weights, self.b2.conv.cn.weights]

        #is this best way to represent weights?
        for neuronWeight in self.c1.full1.neuronRow:
            self.weightMappings.append(neuronWeight.weights)
        for neuronWeight in self.c1.full1.neuronRow:
            self.weightMappings.append(neuronWeight.weights)
        

    def feed_input(self, inputMatrix=np.zeros((64, 400))):
        
        currentMatrix=inputMatrix
        for block in self.blocks:
            currentMatrix=block.forward(currentMatrix)
            self.layerOutputs.extend(block.layerOutputs)
        
        return currentMatrix
    
    #weights are in convolutional_block.conv.cn.weights for convolutional layers
    #in fully connected, they are in: classifier.full1.neuronRow[x].weights
    #and full2.neuronRow[x].weights
    
        

    

    

