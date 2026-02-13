import numpy as np
from networks.atc import atc

def main():
    myCnn=atc()
    
    data = np.load("data/ats_train_spectrograms.npz")
    X = data["X"] 
    
    x_sample = X[0]
    
    ans=myCnn.feedInput(x_sample)
    print(ans)

if __name__ == "__main__":
    main()
    

    