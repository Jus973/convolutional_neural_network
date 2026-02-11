from optimize.im2col import im2col
import numpy as np

if __name__ == "__main__":
    
    testLayer=im2col(kernelSize=3,padding=1, numChannels=1, inputChannels=1)
    inputMatrix=np.array([[[20, 24, 11, 12, 16, 19], 
                          [19, 17, 20, 23, 15, 9],
                          [21, 40, 25, 13, 14, 8],
                          [9, 18, 8, 6, 11, 22],
                          [31, 3, 7, 9, 17, 23],
                          [20, 12, 3, 11, 19, 30]]])
    
    print(testLayer.forward(inputMatrix=inputMatrix))
