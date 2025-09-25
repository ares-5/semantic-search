import json
import uuid
import pandas as pd
import torch
import torch
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from writers.mongo_writer import MongoDBWriter
from writers.elastic_search_writer import ElasticSearchWriter
import classla
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"

nlp_sr = classla.Pipeline('sr', processors='tokenize', use_gpu=torch.cuda.is_available())

# Load original PaSaz dataset (contains both en and sr abstracts)
dataset = pd.read_json("/kaggle/input/pasaz/pasaz.json", orient="records", lines=True)
df = dataset[["title_sr", "title_en", "abstract_en", "abstract_sr", "full_abstract"]]
df = df[df["full_abstract"] == True].reset_index(drop=True)

# --- Load trained models ---
model_en = SentenceTransformer("/ingestion/models/search_model_en")
#model_sr = SentenceTransformer("/ingestion/models/search_model_sr")

tokenizer_en = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-dot-prod-v3")
#tokenizer_sr = AutoTokenizer.from_pretrained("classla/bcms-bertic")

def encode_texts(
        model,
        texts,
        lang='en',
        max_len=512,
        batch_size=32,
        device='cpu', 
        nlp_sr=None):
    """
    Encode a list of texts into embeddings using sentence-level tokenization and pooling.
    - lang: 'en' or 'sr'
    - max_len: max tokens per block (model limit)
    - batch_size: number of texts to process per iteration
    """
    embeddings = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_embs = []

        for text in batch_texts:
            # Sentence-level tokenization
            if lang == 'en':
                sentences = sent_tokenize(text)
            elif lang == 'sr':
                if nlp_sr is None:
                    raise ValueError("nlp_sr pipeline must be provided for Serbian texts")
                doc = nlp_sr(text)
                sentences = [sent.text for sent in doc.sentences]
            else:
                raise ValueError("lang must be 'en' or 'sr'")

            # Group sentences into blocks <= max_len tokens
            blocks = []
            current_block = ""
            for sentence in sentences:
                # Tokenize current block + new sentence
                tokens = model.tokenize((current_block + " " + sentence).strip())
                if len(tokens) <= max_len:
                    current_block = (current_block + " " + sentence).strip()
                else:
                    if current_block:
                        blocks.append(current_block)
                    current_block = sentence
            if current_block:
                blocks.append(current_block)

            # Encode each block
            block_embs = []
            for block in blocks:
                emb = model.encode([block], convert_to_tensor=True, device=device)
                block_embs.append(emb)

            # Pooling: average block embeddings
            text_emb = torch.mean(torch.stack(block_embs), dim=0)
            batch_embs.append(text_emb)

        # Stack batch embeddings
        batch_embs = torch.stack(batch_embs)
        embeddings.append(batch_embs)

    return torch.cat(embeddings, dim=0)

# --- Generate embeddings ---
df["embedding_en"] = encode_texts(
    model_en,
    texts=(df["title_en"] + " " + df["abstract_en"]).tolist(),
    lang = 'en',
    max_len = 512,
    batch_size = 32,
    device = device,
)

# df["embedding_sr"] = encode_texts(
#     model=model_sr,
#     texts=(df["title_sr"] + " " + df["abstract_sr"]).tolist(),
#     lang='sr',
#     max_len=512,
#     batch_size=32,
#     device=device,
#     nlp_sr=nlp_sr
# )

# --- Save embeddings locally ---
with open("embeddings_en.jsonl", "w", encoding="utf-8") as f:
    for row in df.itertuples():
        f.write(json.dumps({
            "text": row.title_en + " " + row.abstract_en,
            "embedding": row.embedding_en.tolist()
        }, ensure_ascii=False) + "\n")

#with open("embeddings_sr.jsonl", "w", encoding="utf-8") as f:
#    for row in df.itertuples():
#        f.write(json.dumps({
#           "text": row.title_sr + " " + row.abstract_sr,
#            "embedding": row.embedding_sr.tolist()
#        }, ensure_ascii=False) + "\n")

# --- Insert into MongoDB ---
if "_id" not in df.columns:
    df["_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

mongo_writer = MongoDBWriter()
df = mongo_writer.insert_docs(df)

# --- Insert into Elasticsearch ---
es_writer = ElasticSearchWriter(
    embedding_dim_en=len(df["embedding_en"][0]),
    #embedding_dim_sr=len(df["embedding_sr"][0]) if "embedding_sr" in df.columns else None
)

# Mapping columns for text indices
text_cols_en = {
    "title": "title_en",
    "details": "abstract_en"
}
text_cols_sr = {
    "title": "title_sr",
    "details": "abstract_sr"
}

# Insert vector embeddings
es_writer.insert_docs(df, lang="en", embedding_col="embedding_en")
#es_writer.insert_docs(df, lang="sr", embedding_col="embedding_sr")

# Insert BM25 text indices
es_writer.insert_docs(df, lang="en")
#es_writer.insert_docs(df, lang="sr")

print("Completed insertion of embeddings and BM25 text indices.")