from fastapi import APIRouter, Query, HTTPException
from models.search_mode import SearchMode
from services import elastic_search_service, mongo_service

router = APIRouter()

@router.get("/")
def search_products(
    query: str = Query(..., description = "Search query"),
    lang: str = Query("en", description = "Language: en | sr"),
    mode: SearchMode = Query(SearchMode.STANDARD, description = "Search mode"),
    size: int = 100,
    candidate_pool: int = 300,
    alpha: float = 0.5
) -> list:
    if mode == SearchMode.STANDARD:
        es_results = elastic_search_service.standard_search(query, lang, size)
        ids = [hit["_id"] for hit in es_results["hits"]["hits"]]

    elif mode == SearchMode.SEMANTIC:
        es_results = elastic_search_service.semantic_search(query, lang, size, candidate_pool)
        ids = [hit["_id"] for hit in es_results["hits"]["hits"]]

    elif mode == SearchMode.HYBRID:
        ids = elastic_search_service.hybrid_search(query, lang, size, alpha)

    elif mode == SearchMode.RERANKED:
        reranked_results = elastic_search_service.reranked_search(query, lang, size, candidate_pool, alpha)
        ids = [cid for cid, _, _ in reranked_results]
    else:
        raise HTTPException(status_code=400, detail = "Invalid search mode")

    results = list(mongo_service.get_dissertations_by_ids(ids))

    return results

@router.get("/index-count")
def get_index_count(
    lang: str = Query("en", description="Language: en | sr"),
    vector: bool = Query(False)
) -> int:
    return elastic_search_service.index_count(lang, vector)
