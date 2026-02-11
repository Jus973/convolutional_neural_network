import numpy as np
from networks.atc import atc

if __name__ == "__main__":
    myCnn=atc()


    data = np.load("data/ats_train_spectrograms.npz")
    X = data["X"] 
    Y = data["y"]

    totalScoreB=0
    currScoreB=0
    #find before accuracy
    for i in range(10):
        x_sample = X[i]
        label = Y[i]

        ans=myCnn.feedInput(x_sample)
        max_index = ans.argmax()
        if (max_index == label):
            currScoreB+=1
        totalScoreB+=1


    myCnn.trainNetwork(numSteps=10)

    
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

    print(currScoreB/totalScoreB)
    print(currScore/totalScore)
    



    
    