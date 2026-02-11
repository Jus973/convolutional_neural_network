import numpy as np

def cross_entropy_loss(p, q):
    #two probability distributions p and q
    #https://www.youtube.com/watch?v=Pwgpl9mKars
    return -sum(p[x] * np.log(q[x]) for x in range(len(p)))


def back_prop(outputV, trueV, cnn, learningRate=0.001):
    #trueV is onehot
    #given one inputV and one outputV shift the cnn's weights. need to run for
    #all data

    #weights are in convolutional_block.conv.cn[x].weights for convolutional layers
    #in fully connected, they are in: classifier.full1.neuronRow[x].weights
    #and full2.neuronRow[x].weights

    upGradient=outputV-trueV
    #dC/dW(L) = dZ(l)/dW(l) * da(L)/dZ(l) * dC/da(L)
    #we know dC/da(L) = error
    
    #prev_layer (size 128)
    #in my curr layer (size 10) each node has 128 weights
    

    def backDenseLayer(theLayer, upGradient):
        for n in range(len(theLayer.neuronRow)):
            theLayer.neuronRow[n].dWeights = upGradient[n] * theLayer.inputMatrix 
            theLayer.neuronRow[n].dBias = upGradient[n]
        
        newError=np.zeros(len(theLayer.inputMatrix))
        for n in range(len(upGradient)):
            newError += upGradient[n]*theLayer.neuronRow[n].weights
        return newError
        
    upGradient = backDenseLayer(cnn.c1.full2, upGradient)
    upGradient = cnn.c1.full2.activationFunction.backward(upGradient) #reverse ReLU
    upGradient = backDenseLayer(cnn.c1.full1, upGradient) #second dense layer


    #undo flatten
    upGradient=upGradient.reshape((64, 8, 50))

    def backConvBlock(theBlock, upGradient):
        #undo maxpool 2d
        kernelSize=theBlock.pool.kernelSize
        stretchedGradient=np.repeat(np.repeat(upGradient, kernelSize, axis=1), 
                                   kernelSize, axis=2)
        stretchedGradient=stretchedGradient*theBlock.pool.mask

        #undo reLU
        upGradient=theBlock.activationFunction.backward(stretchedGradient)
        
        #undo convolution
        downstreamError = np.zeros(theBlock.conv.inputMatrix.shape) 
        convKernelSize=theBlock.conv.kernelSize
        for c in range(len(theBlock.conv.cn)):
            
            filterGradient=np.zeros(theBlock.conv.cn[c].weights.shape)
            flippedFilter = np.flip(theBlock.conv.cn[c].weights)

            for x in range (upGradient.shape[1]):
                for y in range (upGradient.shape[2]):

                    inputPatch = theBlock.conv.inputMatrix[:, x:x+convKernelSize, 
                                                            y:y+convKernelSize]
                    _, p_h, p_w = inputPatch.shape #fix boundary case

                    filterGradient[:, :p_h, :p_w]+=inputPatch*upGradient[c][x][y]
                    
                    errorValue = upGradient[c][x][y]
                    downstreamError[:, x:x+p_h, y:y+p_w] += flippedFilter[:, :p_h, :p_w] * errorValue

            
            theBlock.conv.cn[c].dWeights= filterGradient

        return downstreamError
    
    upGradient=backConvBlock(cnn.b3, upGradient)
    upGradient=backConvBlock(cnn.b2, upGradient)
    upGradient=backConvBlock(cnn.b1, upGradient)
    

    #update all the weights with dWeight and dBias
    #stochastic gradient descent

    #in fully connected, they are in: classifier.full1.neuronRow[x].weights
    #and full2.neuronRow[x].weights

    allWeights=[]
    allWeights.extend(cnn.c1.full1.neuronRow)
    allWeights.extend(cnn.c1.full2.neuronRow)
    allWeights.extend(cnn.b3.conv.cn)
    allWeights.extend(cnn.b2.conv.cn)
    allWeights.extend(cnn.b1.conv.cn)
    
    for eachNeuron in allWeights:
        eachNeuron.weights -= eachNeuron.dWeights * learningRate
        eachNeuron.bias -= eachNeuron.dBias * learningRate    






