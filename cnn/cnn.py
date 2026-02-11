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

        outputSizeX=inputMatrix.shape[1] #TODO inaccurate b/c of stride
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

    def backward(self, upGradient):
        downstreamError = np.zeros(self.inputMatrix.shape) 
        convKernelSize=self.kernelSize
        for c in range(len(self.cn)):
            
            filterGradient=np.zeros(self.cn[c].weights.shape)
            flippedFilter = np.flip(self.cn[c].weights)

            for x in range (upGradient.shape[1]):
                for y in range (upGradient.shape[2]):

                    inputPatch = self.inputMatrix[:, x:x+convKernelSize, 
                                                            y:y+convKernelSize]
                    _, p_h, p_w = inputPatch.shape #fix boundary case

                    filterGradient[:, :p_h, :p_w]+=inputPatch*upGradient[c][x][y]
                    
                    errorValue = upGradient[c][x][y]
                    downstreamError[:, x:x+p_h, y:y+p_w] += flippedFilter[:, :p_h, :p_w] * errorValue

            self.cn[c].dWeights = filterGradient
        
        return downstreamError


class reLU (layer):

    def forward (self, inputMatrix=None):
        rectifiedUnit=lambda x:max(0,x)
        vectorizedUnit=np.vectorize(rectifiedUnit)

        self.inputMatrix=inputMatrix
        self.outputMatrix=vectorizedUnit(inputMatrix)
        return self.outputMatrix
    
    def backward (self, upGradient):
        dInput = upGradient * (self.inputMatrix > 0)
        return dInput

class softmax (layer):
    #expects 1d 
    def forward (self, inputMatrix):
        shift_x = inputMatrix - np.max(inputMatrix)
        exps = np.exp(shift_x)
        self.outputMatrix = exps / np.sum(exps)
        
        return self.outputMatrix
    

    def backward (self, upGradient):
        print("error")
    

class flatten (layer):
    def forward (self, inputMatrix=None):
        self.inputMatrix=inputMatrix
        self.outputMatrix=np.ndarray.flatten(inputMatrix)
        #use self.inputMatrix.shape for calcs
        return self.outputMatrix
    
    def backward (self, upGradient):
        return upGradient.reshape(self.inputMatrix.shape)


class max_pooling_layer(layer):
    def __init__(self, kernelSize=2, stride=2):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
    
    def forward(self, inputMatrix=None): #will be 3 dimensional tuple

        self.mask = np.zeros(inputMatrix.shape)

        #TODO change hardcoded inputMatrix[1] and [2] to something stronger
        numChannels=inputMatrix.shape[0]
        H=inputMatrix.shape[1]
        W=inputMatrix.shape[2]
        outputSizeX=(H - self.kernelSize) // self.stride + 1
        outputSizeY=(W - self.kernelSize) // self.stride + 1
        outputMatrix=np.zeros((numChannels, outputSizeX, outputSizeY)) 
        
        for x in range (0,outputSizeX):
            for y in range (0,outputSizeY):
                
                iX=x*self.stride
                iY=y*self.stride
                
                for i in range(numChannels):
                    
                    patch = inputMatrix[i, iX:iX+self.kernelSize, iY:iY+self.kernelSize]
                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                    outputMatrix[i][x][y] = patch[max_idx]
                    self.mask[i,iX + max_idx[0],iY + max_idx[1]] = 1

        
        self.inputMatrix=inputMatrix
        self.outputMatrix=outputMatrix


        return outputMatrix

    def backward(self, upGradient):
        stretchedGradient=np.repeat(np.repeat(upGradient, self.kernelSize, axis=1), 
                                   self.kernelSize, axis=2)
        stretchedGradient=stretchedGradient*self.mask
        return stretchedGradient


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
    
    def backward(self, upGradient):
        for n in range(len(self.neuronRow)):
            self.neuronRow[n].dWeights = upGradient[n] * self.inputMatrix 
            self.neuronRow[n].dBias = upGradient[n]
        newError=np.zeros(len(self.inputMatrix))
        for n in range(len(upGradient)):
            newError += upGradient[n]*self.neuronRow[n].weights
        return newError
