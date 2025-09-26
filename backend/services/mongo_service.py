from pymongo import MongoClient
from typing import List, Optional
import config

client = MongoClient(config.MONGO_URI)
db = client[config.MONGO_DB]
collection = db[config.MONGO_COLLECTION]

def get_dissertation_by_id(dissertation_id: str):
    return collection.find_one({"_id": dissertation_id})

def get_dissertations_by_ids(dissertation_ids: List[str]):
    if dissertation_ids:
        return list(collection.find({"_id": {"$in": dissertation_ids}}))
    return []

def get_dissertations(skip: int = 0, limit: int = 20):
    return list(collection.find({}).skip(skip).limit(limit))
