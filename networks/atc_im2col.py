from cnn.cnn import *
from optimize.im2col import im2col
import os

class atc_im2col ():

    def __init__ (self, learningRate=0.001):

        self.learningRate=learningRate
        self.layers=[]

        #convolutional blocks
        w1=im2col(3,1,1,16,1)
        self.layers.append(w1)
        self.layers.append(reLU())
        self.layers.append(max_pooling_layer(2, 2))

        w2=im2col(3,1,1,32,16)
        self.layers.append(w2)
        self.layers.append(reLU())
        self.layers.append(max_pooling_layer(2, 2))

        w3=im2col(3,1,1,64,32)
        self.layers.append(w3)
        self.layers.append(reLU())
        self.layers.append(max_pooling_layer(2, 2))

        #classifier
        f1=fully_connected_layer(128,reLU(),25600)
        f2=fully_connected_layer(8,reLU(),128)
        self.layers.append(flatten())
        self.layers.append(f1)
        self.layers.append(reLU())
        self.layers.append(f2)


        self.allWeights=[]

        self.allWeights.extend(w1.cn)
        self.allWeights.extend(w2.cn)
        self.allWeights.extend(w3.cn)

        self.allWeights.extend(f1.neuronRow)
        self.allWeights.extend(f2.neuronRow)

        self.loadWeights()
    
    
    def feedInput(self, inputMatrix):
        currOutput=inputMatrix
        for l in self.layers:
            currOutput=l.forward(currOutput)
        
        self.logits=currOutput
        
        return self.logits
        #probability is softmax().forward(currOutput)

    def updateWeights (self):
        for eachNeuron in self.allWeights:
            eachNeuron.weights -= eachNeuron.dWeights * self.learningRate
            eachNeuron.bias -= eachNeuron.dBias * self.learningRate    

    def loadWeights(self, path="models/atc_weights.npz"):
        if not os.path.exists(path):
            return
        
        data = np.load(path, allow_pickle=True)

        weights = data["weights"]
        biases = data["biases"]

        for neuron, w, b in zip(self.allWeights, weights, biases):
            neuron.weights = w
            neuron.bias = b

    def saveWeights(self, path="models/atc_weights.npz"):
        weights = [neuron.weights for neuron in self.allWeights]
        biases = [neuron.bias for neuron in self.allWeights]

        np.savez(path, weights=np.array(weights, dtype=object), biases=np.array(biases, dtype=object))
    
    def trainNetwork (self, dataPath="data/ats_train_spectrograms.npz", numSteps=10):
        
        data = np.load(dataPath)
        X = data["X"] 
        y = data["y"] 

        if (numSteps==-1):
            numSteps=len(X)
        for i in range(numSteps):
            x_sample = X[i]
            label = y[i]
            
            #build 1 hot label
            label_onehot = np.eye(8)[label]
            self.modular_back_prop(x_sample, label_onehot)
        
        self.saveWeights()

    
    def modular_back_prop(self, inputV, trueV):
        logits=self.feedInput(inputV)
        probs=softmax().forward(logits)
        gradient=probs-trueV
        for layer in self.layers[::-1]:
            gradient=layer.backward(gradient)
        
        self.updateWeights()
    

