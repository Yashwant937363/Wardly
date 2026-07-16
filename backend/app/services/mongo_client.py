import os

from bson import ObjectId
from pymongo import MongoClient
from typing import Any


client = MongoClient(os.getenv("MONGODB_URI"))
db = client["wardly"]
collection = db["wardrobe_items"]


def save_wardrobe_item(document: dict):
    collection.insert_one(document)

def _get_wardrobe_collection():
    """
    Lazily create a single MongoClient for the process lifetime (MongoClient
    is thread-safe and pools connections internally, so it should NOT be
    recreated per call/request).
    """
    return collection

def _mongo_safe(value: Any) -> Any:
    """
    Recursively convert bson.ObjectId (and anything else Mongo might inject,
    like datetimes are fine but ObjectId is not) into JSON-safe types before
    handing a document back to FastAPI's response encoder.
    """
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _mongo_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mongo_safe(v) for v in value]
    return value