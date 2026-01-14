from datasets import load_dataset
import pandas as pd
import numpy as np
import librosa
import io
import soundfile as sf
from PIL import Image


if __name__ == "__main__":
    dataset = load_dataset("jacktol/atc-dataset")
    train_df = dataset["train"].to_pandas()

    # for every sample in train_df (includes audio and text), save a plot in /data for
    # later inspection

    for index, row in train_df.iterrows():
        audio_bytes_io = io.BytesIO(row["audio"]["bytes"])
        y, sr = sf.read(audio_bytes_io, dtype='float32')

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=28,   
                                            n_fft=1024,
                                            hop_length=len(y) // 28)

        S_dB = librosa.power_to_db(S, ref=np.max)

        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        S_pixels = (S_norm * 255).astype(np.uint8)

        img = Image.fromarray(S_pixels)
        img.save(f"data/spectrograms/{row['text']}.png")
        