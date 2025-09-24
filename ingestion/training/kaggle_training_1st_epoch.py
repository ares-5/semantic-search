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

# Cuda setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Seed & Debug
DEBUG = False
torch.manual_seed(42)

# Load dataset
translated_file_path = "/kaggle/input/flipkart-fashion-translated/flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

if DEBUG:
    df = df.sample(100).reset_index(drop=True)
    print("DEBUG mode: using only", len(df), "rows")

# Helpers
def product_row_to_chunks(row, tokenizer, max_length=512, lang="en"):
    """
    Split product info into fields and chunk if longer than max_length.
    Each chunk will be <= max_length tokens.
    """
    if lang == "en":
        parts = {
            "title": f"title: {row['title']}",
            "description": f"description: {row['description']}",
            "details": f"details: {', '.join([f'{list(d.keys())[0]}: {list(d.values())[0]}' for d in row['product_details']])}",
        }
    else:  # sr
        parts = {
            "title": f"title: {row['title_sr']}",
            "description": f"description: {row['description_sr']}",
            "details": f"details: {row['details_sr']}",
        }

    # Common fields
    parts.update({
        "brand": f"brand: {row['brand']}",
        "category": f"category: {row['sub_category']}",
        "seller": f"seller: {row['seller']}"
    })

    chunks = {}
    for key, part in parts.items():
        # tokenize without truncation
        tokens = tokenizer(part, add_special_tokens=True, return_tensors=None)["input_ids"]

        # split into chunks of max_length
        chunk_list = []
        for i in range(0, len(tokens), max_length):
            token_chunk = tokens[i:i + max_length]
            text_chunk = tokenizer.decode(token_chunk, skip_special_tokens=True)
            chunk_list.append(text_chunk.strip())

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
            # pair each chunk of anchor with each chunk of positive
            for chunk_a in chunks_dict[a]:
                for chunk_p in chunks_dict[p]:
                    examples.append(InputExample(texts=[chunk_a, chunk_p]))
    return examples
    
def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples]
    })

# Model initizalization
normalize = models.Normalize()

# EN model
transformer_en = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=512)
pooling_en = models.Pooling(transformer_en.get_word_embedding_dimension(), pooling_mode="mean")
model_en = SentenceTransformer(modules=[transformer_en, pooling_en, normalize]).to(device)
loss_fn_en = losses.CachedMultipleNegativesRankingLoss(model_en, mini_batch_size=32)
tokenizer_en = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# SR model
transformer_sr = models.Transformer("sentence-transformers/distiluse-base-multilingual-cased-v2", max_seq_length=512)
pooling_sr = models.Pooling(transformer_sr.get_word_embedding_dimension(), pooling_mode="mean")
model_sr = SentenceTransformer(modules=[transformer_sr, pooling_sr, normalize]).to(device)
loss_fn_sr = losses.CachedMultipleNegativesRankingLoss(model_sr, mini_batch_size=32)
tokenizer_sr = AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Prepare training data
data_en, data_sr = [], []

for _, row in df.iterrows():
    data_en.extend(generate_input_examples(product_row_to_chunks(row, tokenizer_en, lang="en")))
    data_sr.extend(generate_input_examples(product_row_to_chunks(row, tokenizer_sr, lang="sr")))

dataset_en = input_examples_to_dataset(data_en)
dataset_sr = input_examples_to_dataset(data_sr)

# Training arguments
training_args_en = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-model-en-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

training_args_sr = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/fashion-semantic-model-sr-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

# Train models
trainer_en = SentenceTransformerTrainer(
    model=model_en,
    args=training_args_en,
    train_dataset=dataset_en,
    loss=loss_fn_en
)

trainer_sr = SentenceTransformerTrainer(
    model=model_sr,
    args=training_args_sr,
    train_dataset=dataset_sr,
    loss=loss_fn_sr
)

print("Starting EN training on GPU...")
trainer_en.train()
model_en.save("/kaggle/working/fashion-semantic-model-en")
shutil.make_archive("fashion-semantic-model-en", 'zip', "fashion-semantic-model-en")

print("Starting SR training on GPU...")
trainer_sr.train()
model_sr.save("/kaggle/working/fashion-semantic-model-sr")
shutil.make_archive("fashion-semantic-model-sr", 'zip', "fashion-semantic-model-sr")