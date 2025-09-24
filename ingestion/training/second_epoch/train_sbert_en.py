import json
import pandas as pd
from tqdm import tqdm
import torch
import shutil
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, losses, models
)
from transformers import AutoTokenizer
from datasets import Dataset

# Cuda setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
torch.manual_seed(42)

# Load dataset
translated_file_path = "/kaggle/input/flipkart-fashion-translated/flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

DEBUG = False
if DEBUG:
    df = df.sample(100).reset_index(drop=True)

# Helpers
def product_row_to_chunks(row, tokenizer, max_length=512, lang="en"):
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
        tokens = tokenizer(part, add_special_tokens=True, return_tensors=None)["input_ids"]
        chunk_list = [tokenizer.decode(tokens[i:i+max_length], skip_special_tokens=True).strip() 
                      for i in range(0, len(tokens), max_length)]
        chunks[key] = chunk_list
    return chunks

def generate_input_examples(chunks_dict):
    pairs = [("title","description"),("title","details"),("description","details"),
             ("title","seller"),("description","seller"),("title","brand"),
             ("title","category"),("description","brand"),("description","category")]
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
            query_tokens = tokenizer(item["query"], add_special_tokens=True, return_tensors=None)["input_ids"]
            query_chunks = [tokenizer.decode(query_tokens[i:i+max_length], skip_special_tokens=True).strip()
                            for i in range(0, len(query_tokens), max_length)]
            # chunk positive
            positive_tokens = tokenizer(item["positive"], add_special_tokens=True, return_tensors=None)["input_ids"]
            positive_chunks = [tokenizer.decode(positive_tokens[i:i+max_length], skip_special_tokens=True).strip()
                               for i in range(0, len(positive_tokens), max_length)]
            # create InputExamples for all combinations of chunks
            for q in query_chunks:
                for p in positive_chunks:
                    examples.append(InputExample(texts=[q, p]))
    return examples

# Load model & tokenizer
model = SentenceTransformer("/kaggle/input/fashion-semantic-model-en/transformers/default/1/fashion-semantic-model-en").to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

# Prepare training data
data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    data.extend(generate_input_examples(product_row_to_chunks(row, tokenizer, lang="en")))

# Add synthetic queries
synthetic_file = "/kaggle/input/synthetic-queries-en/synthetic_queries_keyword_en_balanced.jsonl"
data.extend(load_synthetic_examples(synthetic_file, tokenizer))
dataset = input_examples_to_dataset(data)

# Training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-model-en-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    learning_rate=2e-5,
    report_to="none"
)

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss_fn
)

print("Starting EN training...")
trainer.train()
model.save("/kaggle/working/fashion-semantic-model-en")
shutil.make_archive("fashion-semantic-model-en", 'zip', "fashion-semantic-model-en")
