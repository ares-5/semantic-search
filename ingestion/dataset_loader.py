import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, InputExample, models, losses
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm
import os
import json
from transformers import pipeline
from elastic_search_writer import ElasticSearchWriter
from mongo_writer import MongoDBWriter

DEBUG = True

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(torch.cuda.mem_get_info(i))  
    
torch.manual_seed(42)

def product_row_to_chunks(row, tokenizer, max_length=256, lang="en"):
    """Split product info into fields, chunk if too long, return dict of chunks per part.
       Title and description will always be used as an anchor in multiple input combination.
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
        tokens = tokenizer(part, return_tensors=None, truncation=False, add_special_tokens=False)["input_ids"]

        chunk_list = []
        for i in range(0, len(tokens), max_length):
            token_chunk = tokens[i:i + max_length]
            text_chunk = tokenizer.decode(token_chunk, skip_special_tokens=True)
            chunk_list.append(text_chunk.strip())

        chunks[key] = chunk_list

    return chunks

# Prepare training data with multiple anchor-positive combinations
def generate_input_examples(chunks_dict):
    # Define which chunk types to pair as (anchor, positive)
    anchor_positive_pairs = [
        ("title", "description"),
        ("title", "details"),
        ("description", "details"),
        ("title", "seller"),
        ("description", "seller"),
        ("title", "brand"),
        ("title", "category"),
        ("description", "brand"),
        ("description", "category"),
    ]
    
    pairs = []
    for a, p in anchor_positive_pairs:
        if chunks_dict.get(a) and chunks_dict.get(p):
            # use first chunk of each part as anchor and positive (limited based om max_seq_length = 256)
            pairs.append(InputExample(texts=[chunks_dict[a][0], chunks_dict[p][0]]))

    # TODO: Check if synthetic query is needed (query generation)
    # --- Synthetic queries ---
    # product_text = " ".join([chunks_dict.get("title", [""])[0], chunks_dict.get("description", [""])[0]])
    # try:
    #     queries = generate_synthetic_queries(product_text, num_queries=2)
    #     for q in queries:
    #         pairs.append(InputExample(texts=[q, product_text]))
    # except Exception as e:
    #     print("Query generation failed:", e)

    return pairs

failed_rows = []

def safe_translate(text, src="en", dest="sr", retries=3, row_index=None):
    if pd.isna(text) or not str(text).strip():
        return ""

    for i in range(retries):
        try:
            return GoogleTranslator(source=src, target=dest).translate(text)
        except Exception as e:
            print(f"[Row {row_index}] Translation failed ({i+1}/{retries}): {e}")
            time.sleep(2)
    failed_rows.append(row_index)
    return text


def input_examples_to_dataset(input_examples):
    return Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples],
    })

# TODO: Check if synthetic query is needed (query generation)
# def generate_synthetic_queries(text, num_queries=3):
#     """Generate synthetic search queries for a given product text."""
#     queries = query_generator(text, max_length=64, num_return_sequences=num_queries)
#     return [q["generated_text"] for q in queries]

# Dataset containing product data for fashion subdomain of an ecommerce
# https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset?resource=download

# Load dataset (local if exists, else Kaggle + translation)
translated_file_path = "flipkart_fashion_translated.json"
if os.path.exists(translated_file_path):
    print("Loading existing translated dataset...")
    df = pd.read_json(translated_file_path, orient="records", lines=True)
else: 
    print("No translated dataset found, loading raw dataset from Kaggle...")

    # file path containing data for the training
    file_path = "flipkart_fashion_products_dataset.json"

    # Load dataset
    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "aaditshukla/flipkart-fasion-products-dataset",
    file_path,
    #pandas_kwargs={"lines": True}
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    # Shuffle dataset globally to mix categories
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Translate anchors to Serbian
    batch_size = 1000 
    title_sr_list = []

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating title batch {start_idx}-{end_idx}")
        batch_translations = [
            safe_translate(x, row_index=i) 
            for i, x in enumerate(df['title'].iloc[start_idx:end_idx], start=start_idx)
        ]
        title_sr_list.extend(batch_translations)

    df['title_sr'] = pd.Series(title_sr_list)

    # Batch translation for description
    desc_sr_list = []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating description batch {start_idx}-{end_idx}")
        batch_translations = [
            safe_translate(x, row_index=i)
            for i, x in enumerate(df['description'].iloc[start_idx:end_idx], start=start_idx)
        ]
        desc_sr_list.extend(batch_translations)
    df['description_sr'] = pd.Series(desc_sr_list)

    # Batch translation for product details
    details_sr_list = []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        print(f"Translating details batch {start_idx}-{end_idx}")
        batch_translations = [
            safe_translate(
                ", ".join([f"{list(d.keys())[0]}: {list(d.values())[0]}" for d in details]) if details else "",
                row_index=i
            )
            for i, details in enumerate(df['product_details'].iloc[start_idx:end_idx], start=start_idx)
        ]
        details_sr_list.extend(batch_translations)
    df['details_sr'] = pd.Series(details_sr_list)

    # Save failed translations to CSV for later inspection
    if failed_rows:
        pd.Series(failed_rows).to_csv("failed_translations.csv", index=False)
        print(f"Saved {len(failed_rows)} failed translations to failed_translations.csv")

    df.to_json(translated_file_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved translated dataset to {translated_file_path}")

if DEBUG:
    df = df.sample(100).reset_index(drop=True)
    print("DEBUG mode active: using only 100 rows")

# Models initialization
# note - max_seq_length shouldn't be greater than 256. Although it can be increased to 512, it won't lead to better performance
transformer_en = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
pooling_en = models.Pooling(transformer_en.get_word_embedding_dimension(), pooling_mode="mean")

transformer_sr = models.Transformer("DeepPavlov/bert-base-multilingual-cased-sentence", max_seq_length=256)
pooling_sr = models.Pooling(transformer_sr.get_word_embedding_dimension(), pooling_mode="mean")

# normalizer
normalize = models.Normalize()

model_en = SentenceTransformer(modules=[transformer_en, pooling_en, normalize])
model_sr = SentenceTransformer(modules=[transformer_sr, pooling_sr, normalize])

# Dataset preparation for training depending on what we choose as a loss function
# loss function helps out neural network on deciding "weights" - basically a compass for deciding 
# which embeddings are semantically right from neural network point of view
loss_fn_en = losses.CachedMultipleNegativesRankingLoss(model_en, mini_batch_size=16)
loss_fn_sr = losses.CachedMultipleNegativesRankingLoss(model_sr, mini_batch_size=16)

# tokenizer as a helper for correct partitioning of tokens (we want to "cut" words/sentences correctly)
tokenizer_en = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer_sr = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-multilingual-cased-sentence")

# Prepare Training Data
data_en, data_sr = [], []

# TODO: Check if synthetic query is needed (query generation)
#query_generator = pipeline("text2text-generation", model="doc2query/msmarco-t5-base-v1")

for _, row in df.iterrows():
    chunks_en = product_row_to_chunks(row, tokenizer_en, lang="en")
    data_en.extend(generate_input_examples(chunks_en))

    chunks_sr = product_row_to_chunks(row, tokenizer_sr, lang="sr")
    data_sr.extend(generate_input_examples(chunks_sr))

dataset_en = input_examples_to_dataset(data_en)
dataset_sr = input_examples_to_dataset(data_sr)

# Train English Model
training_args_en = SentenceTransformerTrainingArguments(
    output_dir="fashion-semantic-model-en-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

trainer_en = SentenceTransformerTrainer(
    model=model_en,
    args=training_args_en,
    train_dataset=dataset_en,
    loss=loss_fn_en
)

torch.cuda.empty_cache()
print("Cleared GPU cache before training.")

print("Starting en training...")
trainer_en.train()
model_en.save("fashion-semantic-model-en")

# Train Serbian Model
training_args_sr = SentenceTransformerTrainingArguments(
    output_dir="fashion-semantic-model-sr-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

trainer_sr = SentenceTransformerTrainer(
    model=model_sr,
    args=training_args_sr,
    train_dataset=dataset_sr,
    loss=loss_fn_sr
)

torch.cuda.empty_cache()
print("Cleared GPU cache before training.")

print("Starting sr training...")
trainer_sr.train()
model_sr.save("fashion-semantic-model-sr")

# Encode Embeddings for Retrieval
df['text_en_full'] = df.apply(
    lambda row: " | ".join(
        [chunk for sublist in product_row_to_chunks(row, tokenizer_en, lang="en").values() for chunk in sublist]
    ), axis=1
)

embeddings_en = []
for i in tqdm(range(0, len(df['text_en_full']), 128), desc="Encoding EN"):
    batch = df['text_en_full'].iloc[i:i+128].tolist()
    embeddings_en.extend(model_en.encode(batch, batch_size=32, show_progress_bar=False))

df['text_sr_full'] = df.apply(
    lambda row: " | ".join(
        [chunk for sublist in product_row_to_chunks(row, tokenizer_sr, lang="sr").values() for chunk in sublist]
    ), axis=1
)

embeddings_sr = []
for i in tqdm(range(0, len(df['text_sr_full']), 128), desc="Encoding SR"):
    batch = df['text_sr_full'].iloc[i:i+128].tolist()
    embeddings_sr.extend(model_sr.encode(batch, batch_size=32, show_progress_bar=False))

print("Dimension of EN embedding:", len(embeddings_en[0]))
print("Dimension of SR embedding", len(embeddings_sr[0]))

# Save embeddings to .txt
with open("embeddings_en.jsonl", "w", encoding="utf-8") as f:
    for text, emb in zip(df['text_en_full'], embeddings_en):
        f.write(json.dumps({"text": text, "embedding": emb}, ensure_ascii=False) + "\n")

with open("embeddings_sr.jsonl", "w", encoding="utf-8") as f:
    for text, emb in zip(df['text_sr_full'], embeddings_sr):
        f.write(json.dumps({"text": text, "embedding": emb}, ensure_ascii=False) + "\n")


mongo_writer = MongoDBWriter()
df = mongo_writer.insert_docs(df)

es_writer = ElasticSearchWriter(
    embedding_dim_en=len(embeddings_en[0]),
    embedding_dim_sr=len(embeddings_sr[0])
)

# EN && SR embeddings
df["embedding_en"] = embeddings_en
df["embedding_sr"] = embeddings_sr

es_writer.insert_docs(df, index_name="products_en_vector", embedding_col="embedding_en")
es_writer.insert_docs(df, index_name="products_sr_vector", embedding_col="embedding_sr")

# Prepare text_cols mapping for standard inverted indices
text_cols_en = {
    "title": "title",
    "description": "description",
    "details": "product_details",
    "brand": "brand",
    "category": "sub_category",
    "seller": "seller"
}

text_cols_sr = {
    "title": "title_sr",
    "description": "description_sr",
    "details": "details_sr",
    "brand": "brand",
    "category": "sub_category",
    "seller": "seller"
}

es_writer.insert_docs(df, index_name="products_en", text_cols=text_cols_en)
es_writer.insert_docs(df, index_name="products_sr", text_cols=text_cols_sr)
