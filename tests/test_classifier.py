from cnn.cnn import classifier
import numpy as np

if __name__ == "__main__":
    
    testBlock=classifier()
    inputMatrix = np.array([[0.2, 0.4, 0.1, 0.3, 0.5, 0.6],
                            [0.3, 0.2, 0.4, 0.1, 0.5, 0.2],
                            [0.6, 0.7, 0.3, 0.2, 0.1, 0.4],
                            [0.1, 0.5, 0.2, 0.3, 0.4, 0.6],
                            [0.7, 0.1, 0.2, 0.3, 0.5, 0.4],
                            [0.4, 0.3, 0.1, 0.2, 0.5, 0.6]])

    #for test purposes, make weights all equal to each other
    
    for x in testBlock.full1.neuronRow:
        x.weights = np.ndarray.flatten(inputMatrix)
    for x in testBlock.full2.neuronRow:
        x.weights = np.ndarray.flatten(inputMatrix)[:6]
    
    print(testBlock.run(inputMatrix=inputMatrix))
