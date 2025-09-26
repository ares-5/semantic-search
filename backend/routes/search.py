from fastapi import APIRouter, Query, HTTPException
from models.search_mode import SearchMode
from services import elastic_search_service, mongo_service, embeddings

router = APIRouter()

@router.get("/")
def search_products(
    query: str = Query(..., description = "Search query"),
    lang: str = Query("en", description = "Language: en | sr"),
    mode: SearchMode = Query(SearchMode.STANDARD, description = "Search mode"),
    size: int = 10,
    alpha: float = 0.5
) -> list:
    if mode == SearchMode.STANDARD:
        es_results = elastic_search_service.standard_search(query, lang, size)
        ids = [hit["_id"] for hit in es_results["hits"]["hits"]]

    elif mode == SearchMode.SEMANTIC:
        embedding = embeddings.get_embedding(query, lang)
        es_results = elastic_search_service.semantic_search(embedding, lang, size)
        ids = [hit["_id"] for hit in es_results["hits"]["hits"]]

    elif mode == SearchMode.HYBRID:
        embedding = embeddings.get_embedding(query, lang)
        ids = elastic_search_service.hybrid_search(query, embedding, lang, size, alpha)

    else:
        raise HTTPException(status_code=400, detail = "Invalid search mode")
    
    results = list(mongo_service.get_dissertations_by_ids(ids))
    for r in results:
        r["_id"] = str(r["_id"])

    return results

@router.get("/index-count")
def get_index_count(
    lang: str = Query("en", description="Language: en | sr"),
    vector: bool = Query(False)
) -> int:
    return elastic_search_service.index_count(lang, vector)
