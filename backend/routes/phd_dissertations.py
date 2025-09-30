from fastapi import APIRouter, HTTPException, Query, Body
from services import mongo_service
from typing import List, Optional
from models.phd_dissertation import PhdDissertation

router = APIRouter()

@router.get("/{id}", response_model=PhdDissertation)
def get_dissertation_by_id(id: str):
    dissertation = mongo_service.get_dissertation_by_id(dissertation_id=id)
    if not dissertation:
        raise HTTPException(status_code=404, detail="PhD dissertation not found")

    return dissertation

@router.post("/by_ids", response_model=list[PhdDissertation])
def get_dissertations_by_ids(
    dissertation_ids: List[str] = Body(..., embed=True)
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
