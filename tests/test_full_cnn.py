import numpy as np
from cnn.cnn import cnn

if __name__ == "__main__":
    myCnn=cnn()

    data = np.load("data/ats_train_spectrograms.npz")
    X = data["X"] 

    first_spec = X[0, 0] 
    myCnn.feed_input(first_spec)

    for layer in myCnn.layerOutputs:
        print(layer.shape)
    