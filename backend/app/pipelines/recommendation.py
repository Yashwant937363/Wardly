"""
AI outfit-suggestion agent.

Flow:
    query text (e.g. "I want to go to a party")
        -> embed query text with FashionCLIP's TEXT encoder — same 512-dim
           space as the image embeddings already stored on each item, so
           text and image vectors are directly comparable
        -> MongoDB Atlas $vectorSearch, filtered to this owner's closet
           -> shortlist of candidate items
        -> Gemini (structured output) turns the shortlist into 1-2 outfit
           combinations, referencing only items that are actually in the
           shortlist

Requires the "wardrobe_vector_index" MongoDB Atlas Search index (vector
field on vision_metadata.embedding, numDimensions=512, similarity=cosine,
plus a filter field on owner_id) described earlier in this build.
"""

import os
from typing import Any

from bson import ObjectId
from pydantic import BaseModel

from app.models.outfit_suggestions import OutfitSuggestions
from app.services.fashion_clip import embed_query
from app.services.groq_client import get_recommandations as groq_get_recommandation
from app.services.mongo_client import collection



def shortlist_items(query_embedding: list[float], owner_id: str, limit: int = 15) -> list[dict]:
    """
    Vector search MongoDB Atlas for the items closest to the query
    embedding, scoped to one user's wardrobe via the filter field.
    """
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "wardrobe_vector_index",
                "path": "vision_metadata.embedding",
                "queryVector": query_embedding,
                "numCandidates": max(limit * 10, 100),
                "limit": limit,
                "filter": {"owner_id": owner_id},
            }
        },
        {
            "$project": {
                "_id": 1,
                "category": 1,
                "vision_metadata.colors": 1,
                "vision_metadata.detected_attributes": 1,
                "vision_metadata.image.thumbnail_url": 1,
                "fashion_metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))



def build_outfit_suggestions(query: str, shortlisted_items: list[dict]) -> OutfitSuggestions:
    """
    Ask Gemini to build 1-2 outfit combinations from the shortlisted items.
    Gemini is instructed to reference only item_ids present in the
    shortlist; get_outfit_suggestions() double-checks this afterward
    since instruction-following alone isn't a hard guarantee.
    """
    from google import genai

    print("before calling gemini api")
    catalog_for_prompt = [
        {
            "item_id": str(item["_id"]),
            "category": item.get("category"),
            "colors": item.get("vision_metadata", {}).get("colors"),
            "detected_attributes": item.get("vision_metadata", {}).get("detected_attributes"),
            "fashion_metadata": item.get("fashion_metadata"),
        }
        for item in shortlisted_items
    ]

    system_prompt = (
        "Below is a JSON shortlist of items from their closet. Build 1-2 "
        "complete outfit combinations using ONLY these items — do not invent "
        "items that aren't in this list. Reference each chosen item by its "
        "exact item_id. Prefer items whose fashion_metadata occasions/style "
        "match the query, and pick colors that coordinate well together.\n\n"
    )
    user_prompt = (
        f'The user wants outfit ideas for: "{query}"\n\n'
        f"Closet shortlist:\n{catalog_for_prompt}"
    )

    data = groq_get_recommandation(system_prompt=system_prompt, user_prompt=user_prompt)
    print("Got Output")
    return data

def _fetch_missing_image_urls(item_ids: list[str]) -> dict[str, str]:
    """
    Fallback for items whose shortlist projection didn't carry a
    thumbnail_url (e.g. an older document written before that field
    existed). Only hit when a gap is actually detected, and only for
    the specific ids that need it.
    """
    if not item_ids:
        return {}

    cursor = collection.find(
        {"_id": {"$in": [ObjectId(i) for i in item_ids]}},
        {"vision_metadata.image.thumbnail_url": 1},
    )
    return {
        str(doc["_id"]): doc.get("vision_metadata", {})
        .get("image", {})
        .get("thumbnail_url")
        for doc in cursor
    }

# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def get_outfit_suggestions(query: str, owner_id: str) -> OutfitSuggestions:
    print("Function Call")
    query_embedding = embed_query(query)
    print("Got Query Embeddings")
    shortlisted = shortlist_items(query_embedding, owner_id)
    print("done shortlisted")

    if not shortlisted:
        raise ValueError("No wardrobe items found for this user yet.")

    suggestions = build_outfit_suggestions(query, shortlisted)

    # Defensive check: make sure Gemini didn't reference an item_id that
    # wasn't actually in the shortlist (instruction-following isn't a
    # hard guarantee, especially across model/version changes).
    items_by_id = {str(item["_id"]): item for item in shortlisted}
    missing_image_ids: list[str] = []

    for combo in suggestions.combinations:
        for outfit_item in combo.items:
            item = items_by_id.get(outfit_item.item_id)
            if item is None:
                raise ValueError(
                    f"Gemini referenced item_id '{outfit_item.item_id}' "
                    "that wasn't in the shortlist — discarding this response."
                )
            thumb = item.get("vision_metadata", {}).get("image", {}).get("thumbnail_url")
            if thumb:
                outfit_item.image_url = thumb
            else:
                missing_image_ids.append(outfit_item.item_id)

    if missing_image_ids:
        fetched = _fetch_missing_image_urls(missing_image_ids)
        for combo in suggestions.combinations:
            for outfit_item in combo.items:
                if outfit_item.image_url is None and outfit_item.item_id in fetched:
                    outfit_item.image_url = fetched[outfit_item.item_id]

    print(suggestions)
    return suggestions


if __name__ == "__main__":
    query = "I want to go to a party"
    owner_id = "user_123"

    suggestions = get_outfit_suggestions(query, owner_id)
    print(suggestions.model_dump_json(indent=2))