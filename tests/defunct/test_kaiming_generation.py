import numpy as np
from cnn.utils import kaimingMD
from cnn.utils import kaiming

if __name__ == "__main__":

    x=kaiming(5)
    print(x)
    y=kaimingMD((3,5))
    print(y)
    z=kaimingMD((3, 4, 5))
    print(z)
    