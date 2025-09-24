import json
import pandas as pd
from tqdm import tqdm
import torch
import shutil
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, losses
)
from transformers import AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset

# Cuda setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
torch.manual_seed(42)

# Load dataset
translated_file_path = "/kaggle/input/flipkart-fashion-translated/flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

# Split train / eval
train_df = df.sample(frac=0.9, random_state=42)
eval_df = df.drop(train_df.index)

# Helpers
def product_row_to_chunks(row, tokenizer, max_length=512):
    parts = {
        "title": f"title: {row['title']}",
        "description": f"description: {row['description']}",
        "details": f"details: {', '.join([f'{list(d.keys())[0]}: {list(d.values())[0]}' for d in row['product_details']])}",
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
    pairs = [
        ("title","description"),("title","details"),("description","details"),
        ("title","seller"),("description","seller"),("title","brand"),
        ("title","category"),("description","brand"),("description","category")
    ]
    examples = []
    for a, p in pairs:
        if chunks_dict.get(a) and chunks_dict.get(p):
            for ca in chunks_dict[a]:
                for cp in chunks_dict[p]:
                    examples.append(InputExample(texts=[ca, cp]))
    return examples

def input_examples_to_dataset(examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in examples],
        "positive": [ex.texts[1] for ex in examples]
    })

def load_synthetic_examples(file_path, tokenizer, max_length=512):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            # chunk query
            query_tokens = tokenizer(item["query"], add_special_tokens=True)["input_ids"]
            query_chunks = [
                tokenizer.decode(query_tokens[i:i+max_length], skip_special_tokens=True).strip()
                for i in range(0, len(query_tokens), max_length)
            ]
            # chunk positive
            positive_tokens = tokenizer(item["positive"], add_special_tokens=True)["input_ids"]
            positive_chunks = [
                tokenizer.decode(positive_tokens[i:i+max_length], skip_special_tokens=True).strip()
                for i in range(0, len(positive_tokens), max_length)
            ]
            # create InputExamples for all combinations of chunks
            for q in query_chunks:
                for p in positive_chunks:
                    examples.append(InputExample(texts=[q, p]))
    return examples

# Load model & tokenizer
model = SentenceTransformer("/kaggle/input/fashion-semantic-model-en/transformers/default/1/fashion-semantic-model-en").to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

# Prepare train dataset
train_data = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    train_data.extend(generate_input_examples(product_row_to_chunks(row, tokenizer)))

synthetic_file = "/kaggle/input/synthetic-queries-en/synthetic_queries_keyword_en_balanced.jsonl"
train_data.extend(load_synthetic_examples(synthetic_file, tokenizer))
train_dataset = input_examples_to_dataset(train_data)

# Prepare eval dataset
eval_data = []
for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    eval_data.extend(generate_input_examples(product_row_to_chunks(row, tokenizer)))
eval_dataset = input_examples_to_dataset(eval_data)

# Training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-model-en-checkpoint",
    num_train_epochs=3,  # 3 epochs
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    remove_unused_columns=False,
    learning_rate=3e-5,
    metric_for_best_model="eval_loss",
    report_to="none"
)

# Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting EN training...")
trainer.train()

# Save final/best model
model.save("/kaggle/working/fashion-semantic-model-en")
shutil.make_archive("fashion-semantic-model-en", 'zip', "fashion-semantic-model-en")
