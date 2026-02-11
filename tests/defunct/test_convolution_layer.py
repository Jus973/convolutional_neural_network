from cnn.cnn import convolutional_layer
import numpy as np

if __name__ == "__main__":
    
    testLayer=convolutional_layer(kernelSize=3,padding=1)
    inputMatrix=np.array([[20, 24, 11, 12, 16, 19], 
                          [19, 17, 20, 23, 15, 9],
                          [21, 40, 25, 13, 14, 8],
                          [9, 18, 8, 6, 11, 22],
                          [31, 3, 7, 9, 17, 23],
                          [20, 12, 3, 11, 19, 30]])
    
    kernelWeights=np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    
    testLayer.cn.weights = kernelWeights
    
    print(testLayer.run(inputMatrix=inputMatrix))
