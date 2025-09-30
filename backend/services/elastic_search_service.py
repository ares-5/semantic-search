from elasticsearch import Elasticsearch
import torch
import config
from sentence_transformers import CrossEncoder, SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=device)

model_en = SentenceTransformer(config.EN_MODEL_PATH, local_files_only=True)
model_sr = SentenceTransformer(config.SR_MODEL_PATH, local_files_only=True)

es = Elasticsearch(config.ES_HOST)

# Utilities
def get_embedding(query: str, lang: str = "en") -> list[float]:
    model = model_en if lang == "en" else model_sr
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return emb.tolist()

def normalize(scores: dict) -> dict:
    """Min-max normalize scores to [0,1]"""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 1.0 for k in scores}  # all equal
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

# Search functions
def standard_search(query: str, lang: str, size: int):
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "description", "details"]
            }
        }
    }
    return es.search(index=f"phd_dissertations_{lang}", body=body, size=size)

def semantic_search(query: str, lang: str, size: int, candidate_pool: float):
    embedding = get_embedding(query, lang)
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": candidate_pool,
            "num_candidates": candidate_pool
        }
    }
    return es.search(index=f"phd_dissertations_{lang}_vector", body=body)

def hybrid_search(query: str, lang: str, size: int, candidate_pool: int, alpha: float = None):
    """
    Hybrid search combining BM25 and semantic.
    If use_rrf=True, uses Reciprocal Rank Fusion instead of score fusion.
    """
    bm25_result = standard_search(query, lang, size=candidate_pool)
    sem_result = semantic_search(query, lang, size=size, candidate_pool=candidate_pool)

    bm25_scores = {hit["_id"]: hit["_score"] or 0 for hit in bm25_result["hits"]["hits"]}
    sem_scores = {hit["_id"]: hit["_score"] or 0 for hit in sem_result["hits"]["hits"]}

    bm25_norm = normalize(bm25_scores)
    sem_norm = normalize(sem_scores)

    scores = {}
    for pid in set(bm25_scores.keys()).union(sem_scores.keys()):
        b = bm25_norm.get(pid, 0)
        s = sem_norm.get(pid, 0)
        scores[pid] = (1-alpha)*b + alpha*s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:size]
    return [pid for pid, _ in ranked]

def reranked_search(query: str, lang: str, size: int, candidate_pool: int, alpha: float = 0.5):
    """
    Hybrid retrieval + cross-encoder reranking.
    Supports both min-max fusion and RRF.
    """
    candidate_ids = hybrid_search(query, lang, size=candidate_pool, candidate_pool=candidate_pool, alpha=alpha)

    candidates = []
    for cid in candidate_ids:
        res = es.get(index=f"phd_dissertations_{lang}", id=cid)
        doc_text = f"{res['_source']['title']} {res['_source']['details']}"
        candidates.append((cid, doc_text))

    query_doc_pairs = [(query, doc_text) for _, doc_text in candidates]
    scores = reranker.predict(query_doc_pairs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(cid, doc_text, score) for ((cid, doc_text), score) in reranked[:size]]

def index_count(lang: str, vector: bool = False) -> int:
    """Returns number of documents in ES index."""
    if lang not in ("en", "sr"):
        return 0
    index = f"phd_dissertations_{lang}_vector" if vector else f"phd_dissertations_{lang}"
    try:
        if not es.indices.exists(index=index):
            return 0
        return es.count(index=index)["count"]
    except Exception as e:
        print(f"Error accessing index '{index}': {e}")
        return 0
