import numpy as np
from cnn.cnn import cnn

def cross_entropy_loss(p, q):
    #two probability distributions p and q
    #https://www.youtube.com/watch?v=Pwgpl9mKars
    return -sum(p[x] * np.log(q[x]) for x in range(len(p)))


def back_prop(inputV, outputV, trueV, cnn=cnn(), learningRate=0.01):
    #trueV is onehot
    #this is just one example 

    cnn.feed_input(inputV)

    #weights are in convolutional_block.conv.cn.weights for convolutional layers
    #in fully connected, they are in: classifier.full1.neuronRow[x].weights
    #and full2.neuronRow[x].weights

    error=outputV-trueV

    # for x in range (len(error)): 
    #     for y in range (len(prevLayer)):
    #         weights[x][y] -= learningRate * error[i] * x[j]



    

    

