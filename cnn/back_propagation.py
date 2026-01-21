import numpy as np

def cross_entropy_loss(p, q):
    #two probability distributions p and q
    #https://www.youtube.com/watch?v=Pwgpl9mKars
    return -sum(p[x] * np.log(q[x]) for x in range(len(p)))


def back_prop(inputV, outputV, trueV, cnn, learningRate):
    #trueV is onehot
    
    error=outputV-trueV

    

    

