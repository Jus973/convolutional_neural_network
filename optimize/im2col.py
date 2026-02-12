import numpy as np
from cnn.cnn import convolutional_layer
from cnn.utils import kaiming
from cnn.neuron import neuron

class im2col(convolutional_layer): #replace the convolutional layer implementation
    
    def __init__(self, kernelSize=3, stride=1,padding=1,numChannels=1, inputChannels=1):
        super().__init__()
        self.kernelSize=kernelSize
        self.stride=stride
        
        #admittedly a little inefficient but i wanted to keep the same back_prop function
        self.cn=[]
        for _ in range(numChannels):
            self.cn.append(neuron(weights=kaiming(inputChannels*kernelSize*kernelSize)))
        self.padding=padding
        self.numChannels=numChannels
        self.inputChannels=inputChannels


    def forward(self, inputMatrix):
        self.inputMatrix=inputMatrix

        #3d kernel weights * 3d window
        #sum it all together

        paddedMatrix = np.pad(
            inputMatrix,
            ((0,0),
             (self.padding, self.padding),
            (self.padding, self.padding)),
            mode="constant"
        )


        outputSizeX=inputMatrix.shape[1] #assume stride=1
        outputSizeY=inputMatrix.shape[2]
        #outputMatrix=np.zeros((self.numChannels, outputSizeX, outputSizeY)) 
        
        full_col_array = np.zeros((outputSizeX*outputSizeY, self.inputChannels*self.kernelSize*self.kernelSize))

        counter=0
        for x in range (0,outputSizeX):
            for y in range (0,outputSizeY):
                iX=x*self.stride
                iY=y*self.stride
                
                newCol=np.ndarray.flatten(paddedMatrix[:,iX:iX+self.kernelSize,
                                                                iY:iY+self.kernelSize])
                full_col_array[counter]=newCol
                counter+=1
        
        full_weight_array = np.zeros((self.numChannels,self.inputChannels*self.kernelSize*self.kernelSize))
        
        for i in range(len(self.cn)):
            full_weight_array[i]=np.ndarray.flatten(self.cn[i].weights)
        
        outputMatrix=full_col_array@full_weight_array.T
        outputMatrix=outputMatrix.T.reshape(self.numChannels, outputSizeX, outputSizeY)
        
        self.outputMatrix=outputMatrix
        return outputMatrix


