from fastapi import APIRouter, HTTPException, Query
from models.product import Product
from services import mongo_service

router = APIRouter()

@router.get("/{product_id}", response_model=Product)
def get_product(product_id: str):
    product = mongo_service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code = 404, detail = "Product not found")
    
    product["_id"] = str(product["_id"])
    return product

@router.get("/", response_model=list[Product])
def get_catalog(
    category: str = Query(None),
    brand: str = Query(None), 
    skip: int = 0, 
    limit: int = 20
):
    query = {}
    if category:
        query["sub_category"] = category
    if brand:
        query["brand"] = brand
    
    products = mongo_service.get_products(query, skip, limit)
    for p in products:
        p["_id"] = str(p["_id"])
    
    return products