import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "true"

import random
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)

DEVICE = "cpu"
print(f"Device: {DEVICE}")

# -------------------- Load ESCO data --------------------
base = "ESCO dataset - v1.1.1 - classification - en - csv"

occupations = pd.read_csv(
    f"{base}/occupations_en.csv",
    low_memory=False,
)

relations = pd.read_csv(
    f"{base}/occupationSkillRelations_en.csv",
    low_memory=False,
)

skills_df = pd.read_csv(
    f"{base}/skills_en.csv",
    low_memory=False,
)

occupations = occupations.dropna(
    subset=["preferredLabel", "description"]
)

occupations = occupations[
    occupations["description"].str.strip() != ""
]

print(f"Occupations loaded: {len(occupations)}")

# -------------------- Build training pairs --------------------
train_examples = []

# Occupation title and description pairs
for _, row in occupations.iterrows():
    title = str(row["preferredLabel"]).strip()
    desc = str(row["description"]).strip()

    if title and desc:
        train_examples.append(
            InputExample(texts=[title, desc])
        )

        train_examples.append(
            InputExample(texts=[desc, title])
        )

# Occupation and skill pairs
merged = (
    relations.merge(
        occupations[["conceptUri", "preferredLabel"]],
        left_on="occupationUri",
        right_on="conceptUri",
    )
    .merge(
        skills_df[["conceptUri", "preferredLabel"]],
        left_on="skillUri",
        right_on="conceptUri",
        suffixes=("_occ", "_skill"),
    )
)

for _, row in merged.iterrows():
    job = str(row["preferredLabel_occ"]).strip()
    skill = str(row["preferredLabel_skill"]).strip()

    if job and skill:
        train_examples.append(
            InputExample(
                texts=[job, f"requires skill {skill}"]
            )
        )

print(f"Job-skill pairs added: {len(merged)}")
print(f"Total training pairs: {len(train_examples)}")

# -------------------- Train / validation split --------------------
random.seed(42)
random.shuffle(train_examples)

split = int(0.9 * len(train_examples))

train_data = train_examples[:split]
val_data = train_examples[split:]

# Convert training examples to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "sentence1": [
        example.texts[0] for example in train_data
    ],
    "sentence2": [
        example.texts[1] for example in train_data
    ],
})

print(
    f"Train: {len(train_dataset)} | "
    f"Validation: {len(val_data)}"
)

# -------------------- Load model --------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE,
)

# -------------------- Define loss --------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

# -------------------- Training arguments --------------------
args = SentenceTransformerTrainingArguments(
    output_dir="esco-sbert-test",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0,
    use_cpu=True,
    dataloader_pin_memory=False,
    save_strategy="epoch",
    report_to="none",
)

# -------------------- Trainer --------------------
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=train_loss,
    args=args,
)

# -------------------- Start training --------------------
trainer.train()

print("Training completed!")