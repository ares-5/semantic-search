# Semantic Search Evaluation for PaSaz model

import os
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# Config
DATA_JSONL = "hf://datasets/jerteh/PaSaz/PaSaz.jsonl"
MODEL_EN_PATH = "/kaggle/working/search_model_sr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
OUTPUT_FILE = "/kaggle/working/semantic_search_metrics_sr.txt"

# Load dataset
print("Loading dataset...")
df = pd.read_json(DATA_JSONL, lines=True)
df = df[df["full_abstract"] == True].reset_index(drop=True)
queries = df["title_sr"].tolist()
passages = df["abstract_sr"].tolist()
correct_idx = list(range(len(queries)))
n_queries = len(queries)
print(f"Total queries/passages: {n_queries}")

# Load model
print("Loading model...")
model_en = SentenceTransformer(MODEL_EN_PATH, device=DEVICE)

# Evaluation
def compute_metrics(model, queries, passages, correct_idx, top_k=[1, 5, 10], batch_size=32):
    print("Encoding queries...")
    query_embeddings = model.encode(
        queries, convert_to_tensor=True, normalize_embeddings=True,
        device=DEVICE, batch_size=batch_size, show_progress_bar=True
    )
    print("Encoding passages...")
    passage_embeddings = model.encode(
        passages, convert_to_tensor=True, normalize_embeddings=True,
        device=DEVICE, batch_size=batch_size, show_progress_bar=True
    )

    print("Computing cosine similarity matrix...")
    cos_sim_matrix = util.cos_sim(query_embeddings, passage_embeddings)
    recalls = {k: 0 for k in top_k}
    mrr_total = 0.0

    for i in tqdm(range(len(queries)), desc="Calculating metrics"):
        sims = cos_sim_matrix[i]
        sorted_idx = torch.argsort(sims, descending=True)
        correct_rank = (sorted_idx == correct_idx[i]).nonzero(as_tuple=True)[0].item() + 1

        for k in top_k:
            if correct_rank <= k:
                recalls[k] += 1
        mrr_total += 1.0 / correct_rank

    num_queries = len(queries)
    recalls = {k: v / num_queries for k, v in recalls.items()}
    mrr = mrr_total / num_queries
    return recalls, mrr

# Run evaluation
top_k_values = [1, 5, 10]
recalls, mrr = compute_metrics(model_en, queries, passages, correct_idx, top_k=top_k_values)

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as fOut:
    fOut.write("=== Semantic Search Evaluation ===\n")
    fOut.write(f"Recall@1:  {recalls[1]:.4f}\n")
    fOut.write(f"Recall@5:  {recalls[5]:.4f}\n")
    fOut.write(f"Recall@10: {recalls[10]:.4f}\n")
    fOut.write(f"MRR:       {mrr:.4f}\n")

print(f"Evaluation complete! Metrics saved to {OUTPUT_FILE}")
print(f"Recall@1: {recalls[1]:.4f}, Recall@5: {recalls[5]:.4f}, Recall@10: {recalls[10]:.4f}, MRR: {mrr:.4f}")
