from pymongo import MongoClient
from typing import List
import config

client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB]
collection = db[config.MONGO_COLLECTION]

def get_product_by_id(product_id: str):
    """
    Returns product by given product id
    """
    return collection.find_one({"_id": product_id})

def get_products_by_ids(product_ids: List[str]):
    """
    Returns list of products by given product ids
    """
    if len(product_ids) > 0:
        return list(collection.find({"_id": { "$in": product_ids }}))
    else: 
        return list([])

def get_products(query: dict = {}, skip = 0, limit = 20):
    """
    Returns list of products by specified criteria
    """
    return list(collection.find(query).skip(skip).limit(limit))
