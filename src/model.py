import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

MODEL_NAME = "models/wav2vec2-base"


class AIVoiceDetector(nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, waveform):
        outputs = self.wav2vec2(waveform)

        features = outputs.last_hidden_state
        features = features.mean(dim=1)

        logits = self.classifier(features)

        return logits