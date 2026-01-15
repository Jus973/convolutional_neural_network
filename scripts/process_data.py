from datasets import load_dataset
import pandas as pd
import numpy as np
import librosa
import io
import soundfile as sf
from PIL import Image

from scripts.label_dataset import infer_intent

if __name__ == "__main__":  
    dataset = load_dataset("jacktol/ATC-ASR-Dataset")
    train_df = dataset["train"].to_pandas()

    result_list={}
    x_list=[]
    y_list=[]

    for index, row in train_df.iterrows():
        audio_bytes_io = io.BytesIO(row["audio"]["bytes"])
        y, sr = sf.read(audio_bytes_io, dtype='float32')

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64,   
                                            n_fft=1024,
                                            hop_length=160,
                                            power=2.0)

        S_dB = librosa.power_to_db(S, ref=np.max)
        target_frames = 400

        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

        if S_norm.shape[1] < target_frames:
            pad_cols = target_frames - S_norm.shape[1]
            S_fixed = np.pad(S_norm, ((0, 0), (0, pad_cols)), mode='constant')
        else:
            S_fixed = S_norm[:, :target_frames]

        x = S_fixed.astype(np.float32)        # shape (64, 400)
        x = x[np.newaxis, :, :]               # shape (1, 64, 400) -> (C, H, W)

        x_list.append(x)
        label = int(infer_intent(row["text"]))
        y_list.append(label)

        if label in result_list:
            result_list[label]+=1
        else:
            result_list[label]=1
        
        #visualization only
        # S_pixels = (S_fixed * 255).clip(0, 255).astype(np.uint8)
        # img = Image.fromarray(S_pixels)
        # img.save(f"data/spectrograms/{row['text']}.png")
    
    X = np.stack(x_list, axis=0)      
    y = np.array(y_list, dtype=np.int64)  

    np.savez_compressed(
        "data/ats_train_spectrograms.npz",
        X=X,
        y=y,
    )

    print(result_list)