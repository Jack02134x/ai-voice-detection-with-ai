import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import VoiceDataset
from model import AIVoiceDetector


TEST_CSV = "data/test.csv"
MODEL_FILE = "ai_voice_detector.pt"

BATCH_SIZE = 8


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -------------------------
# Dataset
# -------------------------

test_dataset = VoiceDataset(
    TEST_CSV,
    random_crop=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

print(f"Test samples: {len(test_dataset)}")


# -------------------------
# Model
# -------------------------

model = AIVoiceDetector(
    freeze_encoder=True
)

model = model.to(device)


# -------------------------
# Load checkpoint
# -------------------------

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# -------------------------
# Evaluation
# -------------------------

model.eval()

results = []


with torch.inference_mode():

    for index, (waveforms, labels) in enumerate(test_loader):

        waveforms = waveforms.to(
            device,
            non_blocking=True,
        )

        logits = model(waveforms)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = logits.argmax(dim=1)

        # AI probability
        ai_probabilities = probabilities[:, 1]

        # Recover rows belonging to this batch
        start = index * BATCH_SIZE
        end = start + labels.size(0)

        batch_rows = test_dataset.data.iloc[
            start:end
        ]

        for i in range(len(labels)):

            row = batch_rows.iloc[i]

            true_label = labels[i].item()
            prediction = predictions[i].item()
            ai_probability = ai_probabilities[i].item()

            results.append({
                "file": row["file"],
                "label": true_label,
                "prediction": prediction,
                "ai_probability": ai_probability,
                "source": row["source"],
                "speaker": row["speaker"],
                "generator": row["generator"],
                "group": row["group"],
            })


# -------------------------
# Results DataFrame
# -------------------------

results_df = pd.DataFrame(results)

results_df["correct"] = (
    results_df["label"]
    == results_df["prediction"]
)


# -------------------------
# Overall results
# -------------------------

accuracy = results_df["correct"].mean()

human_df = results_df[
    results_df["label"] == 0
]

ai_df = results_df[
    results_df["label"] == 1
]

human_accuracy = human_df["correct"].mean()
ai_accuracy = ai_df["correct"].mean()


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Overall accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Human accuracy:   "
    f"{human_accuracy * 100:.2f}%"
)

print(
    f"AI accuracy:      "
    f"{ai_accuracy * 100:.2f}%"
)

print()
print(
    f"Correct: "
    f"{results_df['correct'].sum()}/{len(results_df)}"
)

print(
    f"Human:   "
    f"{human_df['correct'].sum()}/{len(human_df)}"
)

print(
    f"AI:      "
    f"{ai_df['correct'].sum()}/{len(ai_df)}"
)


# -------------------------
# Human source breakdown
# -------------------------

print()
print("=" * 60)
print("HUMAN SOURCE ACCURACY")
print("=" * 60)

human_sources = (
    human_df
    .groupby("source")["correct"]
    .agg(["sum", "count"])
)

human_sources["accuracy"] = (
    human_sources["sum"]
    / human_sources["count"]
    * 100
)

for source, row in human_sources.iterrows():

    print(
        f"{source:<20} "
        f"{int(row['sum'])}/{int(row['count'])} "
        f"({row['accuracy']:.2f}%)"
    )


# -------------------------
# AI generator breakdown
# -------------------------

print()
print("=" * 60)
print("AI GENERATOR ACCURACY")
print("=" * 60)

ai_generators = (
    ai_df
    .groupby("generator")["correct"]
    .agg(["sum", "count"])
)

ai_generators["accuracy"] = (
    ai_generators["sum"]
    / ai_generators["count"]
    * 100
)

for generator, row in ai_generators.iterrows():

    print(
        f"{generator:<20} "
        f"{int(row['sum'])}/{int(row['count'])} "
        f"({row['accuracy']:.2f}%)"
    )


# -------------------------
# Misclassified samples
# -------------------------

errors = results_df[
    ~results_df["correct"]
].copy()

print()
print("=" * 60)
print("MISCLASSIFIED SAMPLES")
print("=" * 60)

if len(errors) == 0:

    print("No misclassified samples!")

else:

    for _, row in errors.iterrows():

        true_name = (
            "HUMAN"
            if row["label"] == 0
            else "AI"
        )

        predicted_name = (
            "HUMAN"
            if row["prediction"] == 0
            else "AI"
        )

        print()
        print(f"File:       {row['file']}")
        print(f"True:       {true_name}")
        print(f"Predicted:  {predicted_name}")
        print(
            f"AI prob:    "
            f"{row['ai_probability'] * 100:.2f}%"
        )
        print(f"Source:     {row['source']}")

        generator = row["generator"]

        if pd.isna(generator) or generator == "":
            generator = "N/A"

        print(f"Generator:  {generator}")

        print(f"Speaker:    {row['speaker']}")
        print(f"Group:      {row['group']}")


# -------------------------
# Save detailed results
# -------------------------

OUTPUT_FILE = "data/test_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print(f"Detailed results saved to: {OUTPUT_FILE}")