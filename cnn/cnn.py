from cnn.neuron import convNeuron
from cnn.neuron import neuron
from cnn.utils import kaiming
from cnn.utils import kaimingMD
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
    
    def __init__(self, kernelSize=3, stride=1,padding=0,numChannels=16, inputChannels=16):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
        #create numChannel number of convNeurons

        self.cn=[]
        for _ in range(numChannels):
            self.cn.append(convNeuron(weights=kaimingMD((inputChannels, kernelSize, kernelSize)), kernelSize=kernelSize))
        self.padding=padding
        self.numChannels=numChannels
        
    #implementation of convolution formula 
    def forward(self, inputMatrix=None): # should have a num channels for it
        
        paddedMatrix = np.pad(
            inputMatrix,
            ((0,0),
             (self.padding, self.padding),
            (self.padding, self.padding)),
            mode="constant"
        )

        outputSizeX=inputMatrix.shape[1]
        outputSizeY=inputMatrix.shape[2]
        outputMatrix=np.zeros((self.numChannels, outputSizeX, outputSizeY)) 
        
        for x in range (0,outputSizeX):
            for y in range (0,outputSizeY):
                iX=x*self.stride
                iY=y*self.stride
                
                for i in range(self.numChannels):
                    outputMatrix[i][x][y] = self.cn[i].output(paddedMatrix[:,iX:iX+self.kernelSize,
                                                                iY:iY+self.kernelSize])    
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix

        return outputMatrix
    # All neurons in the same convolutional layer share the same weights


class reLU (layer):

    def forward (self, inputMatrix=None):
        rectifiedUnit=lambda x:max(0,x)
        vectorizedUnit=np.vectorize(rectifiedUnit)

        self.inputMatrix=inputMatrix
        self.outputMatrix=vectorizedUnit(inputMatrix)
        return self.outputMatrix
    
    def backward (self, outputMatrix):
        dInput = outputMatrix * (self.inputMatrix > 0)
        return dInput

class softmax (layer):
    #expects 1d 
    def forward (self, inputMatrix=None):
        self.inputMatrix=inputMatrix

        shift_x = inputMatrix - np.max(inputMatrix)
        exps = np.exp(shift_x)
        self.outputMatrix = exps / np.sum(exps)
        
        return self.outputMatrix
    

class flatten (layer):

    def forward (self, inputMatrix=None):
        self.inputMatrix=inputMatrix
        self.outputMatrix=np.ndarray.flatten(inputMatrix)
        return self.outputMatrix

class max_pooling_layer(layer):
    def __init__(self, kernelSize=2, stride=2):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
    
    def forward(self, inputMatrix=None): #will be 3 dimensional tuple
        if inputMatrix.shape[1] % 2 != 0:
            raise ValueError("2x2 maxpool requires %2==0")

        self.mask = np.zeros(inputMatrix.shape)

        #TODO change hardcoded inputMatrix[1] and [2] to something stronger
        numChannels=inputMatrix.shape[0]
        outputSizeX=inputMatrix.shape[1]//self.kernelSize
        outputSizeY=inputMatrix.shape[2]//self.kernelSize
        outputMatrix=np.zeros((numChannels, outputSizeX, outputSizeY)) 
        
        for x in range (0,outputSizeX):
            for y in range (0,outputSizeY):
                
                iX=x*self.stride
                iY=y*self.stride
                
                for i in range(numChannels):
                    outputMatrix[i][x][y] += np.max(inputMatrix[i][iX:iX+self.kernelSize,
                                                        iY:iY+self.kernelSize])
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix

        self.mask=(inputMatrix != 0)

        return outputMatrix

class fully_connected_layer(layer):

    def __init__(self, numNeurons, act_function, inputSize):
        super().__init__()
        self.neuronRow=[]
        self.activationFunction=act_function
        for _ in range(numNeurons):
            self.neuronRow.append(neuron(weights=kaiming(inputSize), act_function=act_function))

    def forward(self, inputMatrix): #input matrix is flattened vector
        outputMatrix=[]
        for i in range(len(self.neuronRow)):
            #represent vector horizontally
            outputMatrix.append(self.neuronRow[i].output(inputMatrix))
        
        self.inputMatrix=inputMatrix
        self.outputMatrix=np.array(outputMatrix)
        return self.outputMatrix



#clean way of combining layers to be blocks while still storing outputs
class block ():
    def __init__(self):
        self.layerOutputs=[]
        self.currentOutput=None

    def performLayer(self, forwardFunction):
        self.currentOutput=forwardFunction(self.currentOutput)
        self.layerOutputs.append(self.currentOutput)

class convolutional_block (block):
    def __init__(self, conv=convolutional_layer(3,1,1,1,1), 
                    activationFunction=reLU(), 
                    pool=max_pooling_layer(2, 2)):
        super().__init__()
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
    def __init__(self, full1=fully_connected_layer(6,reLU(),10),
                        full2=fully_connected_layer(3,reLU(),10)):
        super().__init__()
        self.full1=full1
        self.full2=full2

        #fully_connected_layer.neuronRow[x] are my weights for x neurons
    
    def forward (self, inputMatrix):

        self.currentOutput=inputMatrix

        self.performLayer(flatten().forward)
        self.performLayer(self.full1.forward)
        self.performLayer(self.full2.forward)
        self.performLayer(softmax().forward)

        return self.currentOutput
    

class cnn ():
    def __init__ (self):
        self.b1=convolutional_block(conv=convolutional_layer(3,1,1,16,1), 
                        activationFunction=reLU(), 
                        pool=max_pooling_layer(2, 2))
        self.b2=convolutional_block(conv=convolutional_layer(3,1,1,32,16), 
                        activationFunction=reLU(), 
                        pool=max_pooling_layer(2, 2))
        self.b3=convolutional_block(conv=convolutional_layer(3,1,1,64,32), 
                        activationFunction=reLU(), 
                        pool=max_pooling_layer(2, 2))
        self.c1=classifier(full1=fully_connected_layer(128,reLU(),25600),
                        full2=fully_connected_layer(10,reLU(),128))
        
        self.blocks=[self.b1, self.b2, self.b3, self.c1]
        self.layerOutputs=[] 


        #defunct weight mappings. find way to access it when dealing with backprop
        # self.weightMappings=[self.b1.conv.cn.weights, self.b2.conv.cn.weights]

        # #is this best way to represent weights?
        # for neuronWeight in self.c1.full1.neuronRow:
        #     self.weightMappings.append(neuronWeight.weights)
        # for neuronWeight in self.c1.full1.neuronRow:
        #     self.weightMappings.append(neuronWeight.weights)

    def feed_input(self, inputMatrix=None):
        
        #resize inputMatrix to 3d tuple. originally 1, 64, 400
        inputMatrix=np.expand_dims(inputMatrix, axis=0)

        currentMatrix=inputMatrix
        for block in self.blocks:
            currentMatrix=block.forward(currentMatrix)
            self.layerOutputs.extend(block.layerOutputs)
        
        return currentMatrix
    
    #weights are in convolutional_block.conv.cn.weights for convolutional layers
    #in fully connected, they are in: classifier.full1.neuronRow[x].weights
    #and full2.neuronRow[x].weights

