from fastapi import APIRouter, HTTPException, Query, Body
from services import mongo_service
from typing import List, Optional
from models.phd_dissertation import PhdDissertation

router = APIRouter()

@router.post("/by_ids", response_model=list[PhdDissertation])
def get_dissertations_by_ids(
    dissertation_ids: List[str] = Body(..., embed=True),
    lang: str = "en"
):
    dissertations = mongo_service.get_dissertations_by_ids(dissertation_ids)
    if not dissertations:
        raise HTTPException(status_code=404, detail="PhD dissertations not found")
    
    return dissertations


@router.get("/", response_model=list[PhdDissertation])
def list_dissertations(
    skip: int = 0,
    limit: int = 20,
):
    dissertations = mongo_service.get_dissertations(query, skip, limit)
    return dissertations
