from datasets import load_dataset
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import soundfile as sf


if __name__ == "__main__":
    dataset = load_dataset("jacktol/atc-dataset")
    train_df = dataset["train"].to_pandas()

    for x in train_df:
        print(x)
    input()

    # for every sample in train_df (includes audio and text), save a plot in /data for
    # later inspection

    for index, row in train_df.iterrows():
        audio_bytes_io = io.BytesIO(row["audio"]["bytes"])
        y, sr = sf.read(audio_bytes_io, dtype='float32')

        S = librosa.feature.melspectrogram(y=y, sr=sr, power=2.0)

        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-frequency spectrogram - {row["text"]}')
        plt.tight_layout()
        plt.savefig(f'/data/spectrograms/{index}.png')
        plt.close()
        
        input()
    
    