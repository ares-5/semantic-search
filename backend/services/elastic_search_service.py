from elasticsearch import Elasticsearch
import config

es = Elasticsearch(config.ES_HOST)

def standard_search(query, lang = "en", size = 10):
    """
    Performs search in standard inverted ES index
    """
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "description", "details"]
            }
        }
    }

    return es.search(index=f"phd_dissertations_{lang}", body = body, size = size)

def semantic_search(embedding, lang="en", size=10):
    """
    Performs semantic search using dotProduct similarity via script_score
    """
    body = {
        "size": size,
        "query": {
            "script_score": {
                "query": { "match_all": {} },
                "script": {
                    "source": "dotProduct(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": embedding
                    }
                }
            }
        }
    }

    return es.search(index=f"phd_dissertations_{lang}_vector", body=body)

def hybrid_search(q, embedding, lang="en", size=10, alpha=0.5):
    """
    Combination of standard and semantic search.
    alpha = semantic result weigth (0.0 = only standard, 1.0 = only semantic)
    """
    standard_search_result = standard_search(q, lang, size*2)
    semantic_search_result = semantic_search(embedding, lang, size*2)

    scores = {}

    for hit in standard_search_result["hits"]["hits"]:
        pid = hit["_id"]
        score = hit["_score"] or 0  #BM25 score
        if pid not in scores:
            scores[pid] = 0
        scores[pid] += (1 - alpha) * score

    for hit in semantic_search_result["hits"]["hits"]:
        pid = hit["_id"]
        score = hit["_score"] or 0
        if pid not in scores:
            scores[pid] = 0
        scores[pid] += alpha * score

    # sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:size]
    ids = [pid for pid, _ in ranked]

    return ids

def index_count(lang: str, vector: bool = False) -> int:
    """
    Returns the number of documents in the Elasticsearch index.
    """
    # Ensure lang is valid
    if lang not in ("en", "sr"):
        return 0

    index = f"phd_dissertations_{lang}_vector" if vector else f"phd_dissertations_{lang}"

    # indices.exists expects a string or list of strings
    try:
        if not es.indices.exists(index=index):
            return 0
        count = es.count(index=index)["count"]
        return count
    except Exception as e:
        print(f"Error accessing index '{index}': {e}")
        return 0

