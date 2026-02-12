import numpy as np
from networks.atc_im2col import atc_im2col

if __name__ == "__main__":
    myCnn=atc_im2col()
    

    data = np.load("data/ats_train_spectrograms.npz")
    X = data["X"] 
    Y = data["y"]
    
    totalScore=0
    currScore=0
    #find accuracy
    for i in range(10):
        x_sample = X[i]
        label = Y[i]

        ans=myCnn.feedInput(x_sample)
        max_index = ans.argmax()
        if (max_index == label):
            currScore+=1
        totalScore+=1

        #compare that with label_onehot

    print(currScore/totalScore)
    



    
    