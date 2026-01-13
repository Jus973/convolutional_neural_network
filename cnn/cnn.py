import neuron

class layer ():

    def __init__(self, numNeurons):
        self.neuronRow=[]
        for _ in range(numNeurons):
            self.neuronRow.append(neuron())


class convolutional_layer(layer):

    def __init__(self):
        super.__init__()
    
    # All neurons in the same convolutional layer share the same weights


class cnn ():

    def __init__ (self):
        pass
    
    
    def feed_input(self):
        #input is a tensor of shape (number of inputs x input height x input width x input channels)


        #start the neural network
        pass
    
    

