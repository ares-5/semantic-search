import uuid
import numpy as np
import pandas as pd
import sys
import os

# Add parent folder to path to import writers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from writers.elastic_search_writer import ElasticSearchWriter
from writers.mongo_writer import MongoDBWriter

# --- Load embeddings ---
emb_en = np.load("./embeddings_en.npy")
emb_sr = np.load("./embeddings_sr.npy")

# --- Load dataset ---
dataset = pd.read_json("hf://datasets/jerteh/PaSaz/PaSaz.jsonl", lines=True)
df = dataset[["title_sr", "title_en", "abstract_en", "abstract_sr", "full_abstract"]]
df = df[df["full_abstract"]].reset_index(drop=True)

# --- Assign embeddings to dataframe ---
# Trim embeddings to match dataset length
df["embedding_en"] = list(emb_en[:len(df)])
df["embedding_sr"] = list(emb_sr[:len(df)])

# --- Assign unique IDs if not present ---
if "_id" not in df.columns:
    df["_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

# --- Insert full documents into MongoDB ---
mongo_writer = MongoDBWriter()
df = mongo_writer.insert_docs(df)

print("Embedding EN shape:", len(df["embedding_en"][0]))
print("Embedding SR shape:", len(df["embedding_sr"][0]))

# --- Insert embeddings into Elasticsearch ---
es_writer = ElasticSearchWriter(
    embedding_dim_en=len(df["embedding_en"][0]),
    embedding_dim_sr=len(df["embedding_sr"][0])
)

# Vector embeddings for KNN search
es_writer.insert_docs(df, lang="en", embedding_col="embedding_en")
es_writer.insert_docs(df, lang="sr", embedding_col="embedding_sr")

# BM25 text indices
es_writer.insert_docs(df, lang="en")
es_writer.insert_docs(df, lang="sr")

print("Full documents inserted into MongoDB and embeddings indexed in Elasticsearch.")
