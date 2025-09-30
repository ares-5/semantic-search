# semantic_search_eval_dual.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import random

nltk.download("punkt")

# Config
DATA_JSONL = "hf://datasets/jerteh/PaSaz/PaSaz.jsonl"
MODEL_PATH = "/kaggle/working/model_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
OUTPUT_FILE = "/kaggle/working/semantic_search_eval.txt"

# Load dataset
print("Loading dataset...")
df = pd.read_json(DATA_JSONL, lines=True)
df = df[df["full_abstract"] == True].reset_index(drop=True)

titles = df["title_en"].tolist()
passages = df["abstract_en"].tolist()
n_queries = len(titles)
print(f"Total records: {n_queries}")

# Load model
print("Loading model...")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)


# === Metrics function ===
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


# === Eval 1: Titles as queries ===
print("\nEvaluating with TITLES as queries...")
queries_title = titles
correct_idx = list(range(len(titles)))
recalls_title, mrr_title = compute_metrics(model, queries_title, passages, correct_idx)

# === Eval 2: Abstract snippets as queries ===
print("\nEvaluating with ABSTRACT SNIPPETS as queries...")
queries_snippet = []
correct_idx_snip = []
for i, abs_text in enumerate(passages):
    blocks = sent_tokenize(abs_text)
    blocks = [b for b in blocks if len(b.split()) > 5]  # minimal length
    if not blocks:
        continue
    q = random.choice(blocks)   # pick random snippet
    queries_snippet.append(q)
    correct_idx_snip.append(i)

recalls_snip, mrr_snip = compute_metrics(model, queries_snippet, passages, correct_idx_snip)

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as fOut:
    fOut.write("=== Semantic Search Evaluation ===\n\n")

    fOut.write(">> TITLES as queries:\n")
    fOut.write(f"Recall@1:  {recalls_title[1]:.4f}\n")
    fOut.write(f"Recall@5:  {recalls_title[5]:.4f}\n")
    fOut.write(f"Recall@10: {recalls_title[10]:.4f}\n")
    fOut.write(f"MRR:       {mrr_title:.4f}\n\n")

    fOut.write(">> ABSTRACT SNIPPETS as queries:\n")
    fOut.write(f"Recall@1:  {recalls_snip[1]:.4f}\n")
    fOut.write(f"Recall@5:  {recalls_snip[5]:.4f}\n")
    fOut.write(f"Recall@10: {recalls_snip[10]:.4f}\n")
    fOut.write(f"MRR:       {mrr_snip:.4f}\n")

print("\nEvaluation complete! Metrics saved to", OUTPUT_FILE)

print("\n=== RESULTS ===")
print("Titles as queries:")
print(f"Recall@1: {recalls_title[1]:.4f}, Recall@5: {recalls_title[5]:.4f}, Recall@10: {recalls_title[10]:.4f}, MRR: {mrr_title:.4f}")
print("\nAbstract snippets as queries:")
print(f"Recall@1: {recalls_snip[1]:.4f}, Recall@5: {recalls_snip[5]:.4f}, Recall@10: {recalls_snip[10]:.4f}, MRR: {mrr_snip:.4f}")
