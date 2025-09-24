import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from writers.mongo_writer import MongoDBWriter
from writers.elastic_search_writer import ElasticSearchWriter

# --- Load dataset ---
translated_file_path = "flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

# --- Load trained models ---
model_en = SentenceTransformer("models/fashion-semantic-model-en")
model_sr = SentenceTransformer("models/fashion-semantic-model-sr")

tokenizer_en = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer_sr = AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v2")

# --- Chunking helper ---
def product_row_to_chunks(row, tokenizer, max_length=256, lang="en"):
    """
    Split product info into chunks per field, each chunk <= max_length tokens.
    """
    if lang == "en":
        parts = {
            "title": f"title: {row['title']}",
            "description": f"description: {row['description']}",
            "details": f"details: {', '.join([f'{list(d.keys())[0]}: {list(d.values())[0]}' for d in row['product_details']])}"
        }
    else:
        parts = {
            "title": f"title: {row['title_sr']}",
            "description": f"description: {row['description_sr']}",
            "details": f"details: {row['details_sr']}"
        }

    # common fields
    parts.update({
        "brand": f"brand: {row['brand']}",
        "category": f"category: {row['sub_category']}",
        "seller": f"seller: {row['seller']}"
    })

    chunks = {}
    for key, text in parts.items():
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        chunk_list = [
            tokenizer.decode(tokens[i:i+max_length], skip_special_tokens=True).strip()
            for i in range(0, len(tokens), max_length)
        ]
        chunks[key] = chunk_list

    return chunks

# --- Encode products with pooling ---
def encode_products(df, model, tokenizer, lang="en", batch_size=16):
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {lang}"):
        chunks_dict = product_row_to_chunks(row, tokenizer, lang=lang)
        all_chunks = [chunk for sublist in chunks_dict.values() for chunk in sublist]
        if all_chunks:
            chunk_embeddings = model.encode(all_chunks, batch_size=batch_size, show_progress_bar=False)
            product_embedding = np.mean(chunk_embeddings, axis=0)  # mean pooling
        else:
            product_embedding = np.zeros(model.get_sentence_embedding_dimension())
        embeddings.append(product_embedding)
    return embeddings

# --- Generate embeddings ---
df["embedding_en"] = encode_products(df, model_en, tokenizer_en, lang="en")
df["embedding_sr"] = encode_products(df, model_sr, tokenizer_sr, lang="sr")

# --- Save embeddings locally (optional) ---
with open("embeddings_en.jsonl", "w", encoding="utf-8") as f:
    for row in df.itertuples():
        f.write(json.dumps({"text": row.title + " " + row.description, "embedding": row.embedding_en.tolist()}, ensure_ascii=False) + "\n")

with open("embeddings_sr.jsonl", "w", encoding="utf-8") as f:
    for row in df.itertuples():
        f.write(json.dumps({"text": row.title_sr + " " + row.description_sr, "embedding": row.embedding_sr.tolist()}, ensure_ascii=False) + "\n")

# --- Insert into MongoDB ---
mongo_writer = MongoDBWriter()
df = mongo_writer.insert_docs(df)

# --- Insert into Elasticsearch ---
es_writer = ElasticSearchWriter(
    embedding_dim_en=len(df["embedding_en"][0]),
    embedding_dim_sr=len(df["embedding_sr"][0])
)

# Vector indices
es_writer.insert_docs(df, index_name="products_en_vector", embedding_col="embedding_en")
es_writer.insert_docs(df, index_name="products_sr_vector", embedding_col="embedding_sr")

# Inverted text indices
text_cols_en = {"title": "title", "description": "description", "details": "product_details",
                "brand": "brand", "category": "sub_category", "seller": "seller"}
text_cols_sr = {"title": "title_sr", "description": "description_sr", "details": "details_sr",
                "brand": "brand", "category": "sub_category", "seller": "seller"}

es_writer.insert_docs(df, index_name="products_en", text_cols=text_cols_en)
es_writer.insert_docs(df, index_name="products_sr", text_cols=text_cols_sr)
