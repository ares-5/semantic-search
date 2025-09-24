import json
import pandas as pd
from tqdm import tqdm
import torch
import shutil
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, InputExample, losses
)
from transformers import AutoTokenizer
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
torch.manual_seed(42)

translated_file_path = "/kaggle/input/flipkart-fashion-translated/flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)
DEBUG = False
if DEBUG:
    df = df.sample(100).reset_index(drop=True)

def product_row_to_chunks(row, tokenizer, max_length=512, lang="sr"):
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

def load_synthetic_examples(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            examples.append(InputExample(texts=[item["query"], item["positive"]]))
    return examples

# Load model & tokenizer
model = SentenceTransformer("/kaggle/working/fashion-semantic-model-sr").to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v2")
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

# Prepare training data
data = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    data.extend(generate_input_examples(product_row_to_chunks(row, tokenizer, lang="sr")))

# Add synthetic queries
synthetic_file = "/kaggle/input/flipkart-fashion-synthetic/synthetic_queries_sr.jsonl"
data.extend(load_synthetic_examples(synthetic_file))
dataset = input_examples_to_dataset(data)

training_args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-model-sr-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    learning_rate=2e-5,
    report_to="none"
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=loss_fn
)

print("Starting SR training...")
trainer.train()
model.save("/kaggle/working/fashion-semantic-model-sr")
shutil.make_archive("fashion-semantic-model-sr", 'zip', "fashion-semantic-model-sr")
