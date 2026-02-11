import numpy as np
from cnn.cnn import cnn
from cnn.back_propagation import back_prop
from cnn.cnn import softmax

if __name__ == "__main__":
    myCnn=cnn()

    data = np.load("data/ats_train_spectrograms.npz")
    X = data["X"] 

    first_spec = X[0, 0] 
    output=myCnn.feed_input(first_spec)
    print(output)
    back_prop(output, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], myCnn)
    print(myCnn.feed_input(first_spec))


    
    