from cnn.cnn import max_pooling_layer
import numpy as np

if __name__ == "__main__":
    
    testLayer=max_pooling_layer(kernelSize=2)
    inputMatrix=np.array([[20, 24, 11, 12, 16, 19], 
                          [19, 17, 20, 23, 15, 9],
                          [21, 40, 25, 13, 14, 8],
                          [9, 18, 8, 6, 11, 22],
                          [31, 3, 7, 9, 17, 23],
                          [20, 12, 3, 11, 19, 30]])
    
    print(testLayer.run(inputMatrix=inputMatrix))
