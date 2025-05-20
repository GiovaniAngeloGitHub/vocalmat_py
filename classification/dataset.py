import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch import tensor, long
from PIL import Image
import io

class USVDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.annotations.iloc[idx]['audio_file'])
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Convert spectrogram to image
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        plt.axis('off')
        librosa.display.specshow(S_dB, sr=sr)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        buf.close()
        plt.close(fig)

        label = int(self.annotations.iloc[idx]['label'])
        label = tensor(label, dtype=long)

        if self.transform:
            image = self.transform(image)

        return image, label
