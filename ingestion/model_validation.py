import torch
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load fine-tuned model
sbert_model = SentenceTransformer("/kaggle/working/search_model_en").to(device)

# Load generated synthetic dataset
out_file = "/kaggle/working/generated_queries_all.tsv"
queries, passages, correct_idx = [], [], []

with open(out_file, "r", encoding="utf-8") as fIn:
    for idx, line in enumerate(fIn):
        try:
            query, passage = line.strip().split("\t", maxsplit=1)
            queries.append(query)
            passages.append(passage)
            correct_idx.append(idx)
        except:
            continue

# Sliding window + mean pooling for text longer than 512
def encode_long_texts(model, texts, max_len=512, stride=128, batch_size=32):
    all_embeddings = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_embs = []
        for text in batch_texts:
            tokens = model.tokenizer.encode(text)
            if len(tokens) > max_len:
                seg_embs = []
                for i in range(0, len(tokens), max_len - stride):
                    seg_tokens = tokens[i:i+max_len]
                    seg_text = model.tokenizer.decode(seg_tokens)
                    seg_emb = model.encode([seg_text], convert_to_tensor=True, device=device)
                    seg_embs.append(seg_emb)
                seg_embs = torch.cat(seg_embs, dim=0)
                seg_embs = torch.mean(seg_embs, dim=0, keepdim=True)  # mean pooling
                batch_embs.append(seg_embs)
            else:
                batch_embs.append(model.encode([text], convert_to_tensor=True, device=device))
        batch_embs = torch.cat(batch_embs, dim=0)
        all_embeddings.append(batch_embs)
    return torch.cat(all_embeddings, dim=0)

# Compute Recall@k i MRR
def compute_metrics(model, queries, passages, correct_idx, top_k=[1,5,10], batch_size=32):
    query_embeddings = encode_long_texts(model, queries, batch_size=batch_size)
    passage_embeddings = encode_long_texts(model, passages, batch_size=batch_size)

    cos_sim_matrix = util.cos_sim(query_embeddings, passage_embeddings)
    recalls = {k: 0 for k in top_k}
    mrr_total = 0.0

    for i in range(len(queries)):
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

# Run metrics
top_k_values = [1,5,10]
recalls, mrr = compute_metrics(sbert_model, queries, passages, correct_idx, top_k=top_k_values)
print(f"Recall@k: {recalls}")
print(f"MRR: {mrr:.4f}")

# Inspect a few examples
for i in range(5):
    top_match_idx = torch.argmax(util.cos_sim(
        encode_long_texts(sbert_model, [queries[i]]),
        encode_long_texts(sbert_model, passages)
    )).item()
    print(f"\nQuery: {queries[i]}")
    print(f"Top matched passage: {passages[top_match_idx]}")
