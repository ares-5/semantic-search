import json
import numpy as np
from elasticsearch import Elasticsearch
import os

import config
from services import elastic_search_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "search_queries.json")

with open(json_path, "r", encoding="utf-8") as f:
    search_queries = json.load(f)

K = 10  # top 10
es = Elasticsearch(config.ES_HOST)

# Metric helpers
def precision_at_k(pred_ids, gold_ids, k):
    """
    Precision@K (P@K)

    Definicija:
        Mera preciznosti u top-K rezultata pretrage.
        Pokazuje koliki deo dokumenata u prvih K koje model vrati
        zapravo jeste relevantan.

    Formula:
        P@K = (broj relevantnih dokumenata u top-K) / K

    Parametri:
        pred_ids (list): Lista ID-jeva dokumenata koje je sistem vratio, poređani po rangiranju.
        gold_ids (list): Lista stvarno relevantnih dokumenata (ground truth).
        k (int): Broj top rezultata koje posmatramo.

    Vraća:
        float: Preciznost u intervalu [0, 1].

    Primer:
        Ako je K=10 i 6 od prvih 10 dokumenata relevantno → P@10 = 0.6
    """
    return len(set(pred_ids[:k]) & set(gold_ids)) / k

def recall_at_k(pred_ids, gold_ids, k):
    """
    Recall@K (R@K)

    Definicija:
        Mera pokrivenosti relevantnih dokumenata u top-K rezultata.
        Pokazuje koliki deo svih relevantnih dokumenata je sistem uspeo da pronađe.

    Formula:
        R@K = (broj relevantnih dokumenata u top-K) / (ukupan broj relevantnih dokumenata)

    Parametri:
        pred_ids (list): Lista ID-jeva dokumenata koje je sistem vratio.
        gold_ids (list): Lista stvarno relevantnih dokumenata.
        k (int): Broj top rezultata koje posmatramo.

    Vraća:
        float: Recall u intervalu [0, 1].

    Primer:
        Ako postoji ukupno 7 relevantnih dokumenata, a model je u top-10 pronašao 6 → R@10 = 6/7 ≈ 0.86
    """
    return len(set(pred_ids[:k]) & set(gold_ids)) / len(gold_ids) if gold_ids else 0

def ndcg_at_k(pred_ids, gold_ids, k):
    """
    Normalized Discounted Cumulative Gain (nDCG@K)

    Definicija:
        Mera kvaliteta rangiranja rezultata.
        Ne gleda samo da li su dokumenti relevantni, već i na kojoj su poziciji.
        Relevantni dokumenti bliže vrhu liste doprinose više nego oni na dnu.

    Formula:
        DCG = Σ (rel_i / log2(i+2)),  i = pozicija dokumenta u listi
        IDCG = maksimalni mogući DCG ako su svi relevantni na vrhu
        nDCG = DCG / IDCG

    Parametri:
        pred_ids (list): Lista ID-jeva dokumenata koje je sistem vratio.
        gold_ids (list): Lista stvarno relevantnih dokumenata.
        k (int): Broj top rezultata koje posmatramo.

    Vraća:
        float: nDCG vrednost u intervalu [0, 1].
               1 znači savršeno rangiranje (svi relevantni na vrhu),
               0 znači nijedan relevantan u top-K.

    Primer:
        Ako su relevantni dokumenti raspoređeni visoko u listi, nDCG@10 može biti npr. 0.89.
    """
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

for r in results:
    print(f"\nQuery: {r['query']}")
    for method in ["bm25", "semantic", "hybrid", "reranked"]:
        print(f"  {method:<9} -> P@10={r[method]['p@10']:.2f}, "
              f"R@10={r[method]['r@10']:.2f}, "
              f"nDCG@10={r[method]['ndcg@10']:.2f}")
