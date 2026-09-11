from dataset import VoiceDataset


dataset = VoiceDataset("data/dataset.csv")

print(f"Dataset size: {len(dataset)}")

for i in range(len(dataset)):
    waveform, label = dataset[i]

    print()
    print(f"Sample {i}")
    print(f"Audio samples: {waveform.shape[0]}")
    print(f"Duration: {waveform.shape[0] / 16_000:.2f} seconds")
    print(f"Label: {label.item()}")