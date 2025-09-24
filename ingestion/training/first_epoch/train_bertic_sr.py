import pandas as pd
from tqdm import tqdm
import torch
import shutil
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, models, losses
)
from transformers import AutoTokenizer
from datasets import Dataset

# CUDA setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)

# Load dataset
translated_file_path = "/kaggle/input/flipkart-fashion-translated/flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

# Helpers
def product_row_to_chunks(row, tokenizer, max_length=512):
    parts = {
        "title": f"title: {row['title_sr']}",
        "description": f"description: {row['description_sr']}",
        "details": f"details: {row['details_sr']}",
        "brand": f"brand: {row['brand']}",
        "category": f"category: {row['sub_category']}",
        "seller": f"seller: {row['seller']}"
    }
    chunks = {}
    for key, part in parts.items():
        tokens = tokenizer(part, add_special_tokens=True)["input_ids"]
        chunk_list = [
            tokenizer.decode(tokens[i:i+max_length], skip_special_tokens=True).strip()
            for i in range(0, len(tokens), max_length)
        ]
        chunks[key] = chunk_list
    return chunks

def generate_input_examples(chunks_dict):
    anchor_positive_pairs = [
        ("title", "description"), ("title", "details"), ("description", "details"),
        ("title", "seller"), ("description", "seller"), ("title", "brand"),
        ("title", "category"), ("description", "brand"), ("description", "category")
    ]
    examples = []
    for a, p in anchor_positive_pairs:
        if chunks_dict.get(a) and chunks_dict.get(p):
            for chunk_a in chunks_dict[a]:
                for chunk_p in chunks_dict[p]:
                    examples.append(InputExample(texts=[chunk_a, chunk_p]))
    return examples

def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples]
    })

# Tokenizer & model
normalize = models.Normalize()
transformer = models.Transformer("classla/bcms-bertic", max_seq_length=512)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
model = SentenceTransformer(modules=[transformer, pooling, normalize]).to(device)
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic")

# Build dataset
data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    data.extend(generate_input_examples(product_row_to_chunks(row, tokenizer)))

dataset = input_examples_to_dataset(data)

# Training args
training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-bertic-model-sr-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss_fn
)

print("Starting SR training...")
trainer.train()
model.save("/kaggle/working/fashion-semantic-bertic-model-sr")
shutil.make_archive("fashion-semantic-bertic-model-sr", 'zip', "fashion-semantic-bertic-model-sr")