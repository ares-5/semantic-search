import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load fine-tuned model
sbert_model = SentenceTransformer("/kaggle/input/final_en_model/transformers/default/1").to(device)

# Load synthetic dataset
out_file = "/kaggle/input/queries/generated_queries_all (3).tsv"
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

# Sliding window + mean pooling for long texts
def encode_long_texts(model, texts, max_len=512, stride=128, batch_size=32):
    all_embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size), disable=True):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_embs = []
        for text in batch_texts:
            tokens = model.tokenizer.encode(text)
            if len(tokens) > max_len:
                seg_embs = []
                for i in range(0, len(tokens), max_len - stride):
                    seg_tokens = tokens[i:i+max_len]
                    seg_text = model.tokenizer.decode(seg_tokens)
                    seg_emb = model.encode([seg_text], convert_to_tensor=True, device=device, show_progress_bar=False)
                    seg_embs.append(seg_emb)
                seg_embs = torch.cat(seg_embs, dim=0)
                seg_embs = torch.mean(seg_embs, dim=0, keepdim=True)
                batch_embs.append(seg_embs)
            else:
                batch_embs.append(model.encode([text], convert_to_tensor=True, device=device))
        batch_embs = torch.cat(batch_embs, dim=0)
        all_embeddings.append(batch_embs)
    return torch.cat(all_embeddings, dim=0)

# Compute Recall@k and MRR
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

# Run evaluation
top_k_values = [1,5,10]
recalls, mrr = compute_metrics(sbert_model, queries, passages, correct_idx, top_k=top_k_values)

print("=== Semantic Search Evaluation ===")
print(f"Recall@1:  {recalls[1]:.4f}")
print(f"Recall@5:  {recalls[5]:.4f}")
print(f"Recall@10: {recalls[10]:.4f}")
print(f"MRR:       {mrr:.4f}")
