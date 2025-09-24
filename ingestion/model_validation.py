from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
translated_file_path = "flipkart_fashion_translated.json"
df = pd.read_json(translated_file_path, orient="records", lines=True)

# Sample validation set
val_sample_size = 2000
val_df = df.sample(val_sample_size, random_state=42).reset_index(drop=True)

# Load models
model_en = SentenceTransformer("models/fashion-semantic-model-en").to(device)
model_sr = SentenceTransformer("models/fashion-semantic-model-sr").to(device)

# Helper: generate realistic query variations
def generate_queries(row, lang="en"):
    """
    Create multiple query variations from product title
    """
    queries = []
    title = row['title'] if lang=="en" else row['title_sr']
    # Full title
    queries.append(title)
    # First 2-3 words
    queries.append(" ".join(title.split()[:3]))
    # Last 1-2 words
    queries.append(" ".join(title.split()[-2:]))
    # Capitalization / lowercase variants
    queries.append(title.lower())
    queries.append(title.upper())
    return queries

# Prepare queries and correct indices
queries_en, correct_idx_en = [], []
queries_sr, correct_idx_sr = [], []
product_texts_en, product_texts_sr = [], []

product_ids = list(val_df.index)  # just use row index as ID for testing

for idx, row in val_df.iterrows():
    q_en = generate_queries(row, "en")
    queries_en.extend(q_en)
    correct_idx_en.extend([idx] * len(q_en))

    q_sr = generate_queries(row, "sr")
    queries_sr.extend(q_sr)
    correct_idx_sr.extend([idx] * len(q_sr))

    # Product text for semantic search: description + details
    details_en = ", ".join([f"{list(d.keys())[0]}: {list(d.values())[0]}" for d in row['product_details']])
    product_texts_en.append(row['description'] + " " + details_en)

    details_sr = row['details_sr'] if 'details_sr' in row else ""
    product_texts_sr.append(row['description_sr'] + " " + details_sr)

# Metrics computation
def compute_metrics(model, queries, product_texts, correct_idx, top_k=[1,5,10], batch_size=32):
    # Encode queries and products
    query_embeddings = model.encode(queries, convert_to_tensor=True, batch_size=batch_size, device=device)
    product_embeddings = model.encode(product_texts, convert_to_tensor=True, batch_size=batch_size, device=device)

    cos_sim_matrix = util.cos_sim(query_embeddings, product_embeddings)

    recalls = {k: 0 for k in top_k}
    mrr_total = 0.0

    for i in range(len(queries)):
        sims = cos_sim_matrix[i]
        sorted_idx = torch.argsort(sims, descending=True)
        correct_rank = (sorted_idx == correct_idx[i]).nonzero(as_tuple=True)[0].item() + 1  # ranks start at 1

        for k in top_k:
            if correct_rank <= k:
                recalls[k] += 1
        mrr_total += 1.0 / correct_rank

    num_queries = len(queries)
    recalls = {k: v / num_queries for k, v in recalls.items()}
    mrr = mrr_total / num_queries
    return recalls, mrr

# Run metrics for English
top_k_values = [1,5,10]
print("Evaluating English model...")
recalls_en, mrr_en = compute_metrics(model_en, queries_en, product_texts_en, correct_idx_en, top_k=top_k_values)
print(f"Recall@k (EN): {recalls_en}")
print(f"MRR (EN): {mrr_en:.4f}")

# Run metrics for Serbian
print("\nEvaluating Serbian model...")
recalls_sr, mrr_sr = compute_metrics(model_sr, queries_sr, product_texts_sr, correct_idx_sr, top_k=top_k_values)
print(f"Recall@k (SR): {recalls_sr}")
print(f"MRR (SR): {mrr_sr:.4f}")

# Optional: inspect a few examples
for i in range(5):
    top_match_idx = torch.argmax(util.cos_sim(model_en.encode([queries_en[i]], convert_to_tensor=True, device=device),
                                              model_en.encode(product_texts_en, convert_to_tensor=True, device=device))).item()
    print(f"\nQuery: {queries_en[i]}")
    print(f"Top matched product: {product_texts_en[top_match_idx]}")
