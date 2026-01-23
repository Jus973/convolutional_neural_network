import numpy as np

#1d
def kaiming(size): #kaiming initialization. std sqrt2/size, mean of 0
    std_dev=np.sqrt(2. / size)
    weights=np.random.normal(loc=0, scale=std_dev, size=(size))
    return weights


#multi dimensional
def kaimingMD(shape):
    std_dev=np.sqrt(2 / np.prod(shape[1:])) # should be # of inputs to one neuron
    weights=np.random.normal(loc=0, scale=std_dev, size=shape)
    return weights
