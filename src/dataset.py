import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset


TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 4  # seconds
CLIP_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION


class VoiceDataset(Dataset):
    def __init__(self, csv_file, random_crop=True):
        self.data = pd.read_csv(csv_file)
        self.random_crop = random_crop

    def __len__(self):
        return len(self.data)

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample → 16 kHz
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def prepare_clip(self, waveform):
        length = waveform.shape[0]

        if length > CLIP_SAMPLES:

            if self.random_crop:
                # Random crop for training
                start = torch.randint(
                    0,
                    length - CLIP_SAMPLES + 1,
                    (1,),
                ).item()
            else:
                # Deterministic crop for validation/test
                start = 0

            waveform = waveform[
                start:start + CLIP_SAMPLES
            ]

        elif length < CLIP_SAMPLES:

            padding = CLIP_SAMPLES - length

            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding),
            )

        # Normalize waveform
        waveform = waveform - waveform.mean()

        std = waveform.std()

        if std > 0:
            waveform = waveform / std

        return waveform

    def __getitem__(self, index):
        row = self.data.iloc[index]

        waveform = self.load_audio(row["file"])
        waveform = self.prepare_clip(waveform)

        label = torch.tensor(
            row["label"],
            dtype=torch.long,
        )

        return waveform, label