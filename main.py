from datasets import load_dataset
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = load_dataset("jacktol/atc-dataset")
    train_df = dataset["train"].to_pandas()

    print(type(dataset["train"])) 

    for index, row in train_df.iterrows():
        print(f"Row {index}: {row.keys()}") #we see the data type as 'audio' and 'text'
        print(type(row['audio'])) 
        input()
    
    