import json
import numpy as np
from elasticsearch import Elasticsearch
import os

from backend import config
from backend.services import elastic_search_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder gde je evaluate.py
json_path = os.path.join(BASE_DIR, "search_queries.json")

with open(json_path, "r", encoding="utf-8") as f:
    search_queries = json.load(f)

K = 10  # top 10
es = Elasticsearch(config.ES_HOST)

# Metric helpers
def precision_at_k(pred_ids, gold_ids, k):
    return len(set(pred_ids[:k]) & set(gold_ids)) / k

def recall_at_k(pred_ids, gold_ids, k):
    return len(set(pred_ids[:k]) & set(gold_ids)) / len(gold_ids) if gold_ids else 0

def ndcg_at_k(pred_ids, gold_ids, k):
    dcg = 0.0
    for i, pid in enumerate(pred_ids[:k]):
        if pid in gold_ids:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(gold_ids), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

results = []
for entry in search_queries:
    query = entry["query"]
    lang = entry["lang"]
    gold_ids = entry["relevant_ids"]

    # 1. BM25
    bm25 = elastic_search_service.standard_search(query, lang, size=K)
    bm25_ids = [hit["_id"] for hit in bm25["hits"]["hits"]]

    # 2. Semantic
    semantic = elastic_search_service.semantic_search(query, lang, size=K, candidate_pool=300)
    sem_ids = [hit["_id"] for hit in semantic["hits"]["hits"]]

    # 3. Hybrid
    hybrid_ids = elastic_search_service.hybrid_search(query, lang, size=K, candidate_pool=300, alpha=0.6)

    # 4. Reranked
    reranked = elastic_search_service.reranked_search(query, lang, size=K, candidate_pool=300, alpha=0.6)
    reranked_ids = [cid for cid, _, _ in reranked]

    # Izračunaj metrike
    metrics = {
        "query": query,
        "bm25": {
            "p@10": precision_at_k(bm25_ids, gold_ids, K),
            "r@10": recall_at_k(bm25_ids, gold_ids, K),
            "ndcg@10": ndcg_at_k(bm25_ids, gold_ids, K),
        },
        "semantic": {
            "p@10": precision_at_k(sem_ids, gold_ids, K),
            "r@10": recall_at_k(sem_ids, gold_ids, K),
            "ndcg@10": ndcg_at_k(sem_ids, gold_ids, K),
        },
        "hybrid": {
            "p@10": precision_at_k(hybrid_ids, gold_ids, K),
            "r@10": recall_at_k(hybrid_ids, gold_ids, K),
            "ndcg@10": ndcg_at_k(hybrid_ids, gold_ids, K),
        },
        "reranked": {
            "p@10": precision_at_k(reranked_ids, gold_ids, K),
            "r@10": recall_at_k(reranked_ids, gold_ids, K),
            "ndcg@10": ndcg_at_k(reranked_ids, gold_ids, K),
        },
    }
    results.append(metrics)

# Ispiši pregledno
for r in results:
    print(f"\nQuery: {r['query']}")
    for method in ["bm25", "semantic", "hybrid", "reranked"]:
        print(f"  {method:<9} -> P@10={r[method]['p@10']:.2f}, "
              f"R@10={r[method]['r@10']:.2f}, "
              f"nDCG@10={r[method]['ndcg@10']:.2f}")
