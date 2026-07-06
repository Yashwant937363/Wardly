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

from wardrobe_image_processor import _get_wardrobe_collection, _load_fashion_clip

# ----------------------------------------------------------------------
# Query embedding (text side of the FashionCLIP joint embedding space)
# ----------------------------------------------------------------------

_fashion_clip_tokenizer: Any = None


def _load_fashion_clip_tokenizer() -> Any:
    global _fashion_clip_tokenizer
    if _fashion_clip_tokenizer is None:
        import open_clip

        _fashion_clip_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")
    return _fashion_clip_tokenizer


def embed_query(query: str) -> list[float]:
    """
    Embed a text query into the SAME space as the stored image embeddings.
    """
    import torch

    model, _ = _load_fashion_clip()
    tokenizer = _load_fashion_clip_tokenizer()

    tokens = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(tokens, normalize=True)

    return text_features[0].tolist()


# ----------------------------------------------------------------------
# Vector search shortlist
# ----------------------------------------------------------------------

def shortlist_items(query_embedding: list[float], owner_id: str, limit: int = 15) -> list[dict]:
    """
    Vector search MongoDB Atlas for the items closest to the query
    embedding, scoped to one user's wardrobe via the filter field.
    """
    collection = _get_wardrobe_collection()
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


# ----------------------------------------------------------------------
# Gemini: turn the shortlist into outfit combinations
# ----------------------------------------------------------------------

class OutfitItemRef(BaseModel):
    item_id: str
    category: str
    reason: str
    image_url: str | None = None


class OutfitCombination(BaseModel):
    title: str
    items: list[OutfitItemRef]
    styling_notes: str


class OutfitSuggestions(BaseModel):
    combinations: list[OutfitCombination]


def build_outfit_suggestions(query: str, shortlisted_items: list[dict]) -> OutfitSuggestions:
    """
    Ask Gemini to build 1-2 outfit combinations from the shortlisted items.
    Gemini is instructed to reference only item_ids present in the
    shortlist; get_outfit_suggestions() double-checks this afterward
    since instruction-following alone isn't a hard guarantee.
    """
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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

    prompt = (
        f'The user wants outfit ideas for: "{query}"\n\n'
        "Below is a JSON shortlist of items from their closet. Build 1-2 "
        "complete outfit combinations using ONLY these items — do not invent "
        "items that aren't in this list. Reference each chosen item by its "
        "exact item_id. Prefer items whose fashion_metadata occasions/style "
        "match the query, and pick colors that coordinate well together.\n\n"
        f"Closet shortlist:\n{catalog_for_prompt}"
    )

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[{"type": "text", "text": prompt}],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": OutfitSuggestions.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
    )

    if interaction.output_text is None:
        raise RuntimeError("Gemini returned no output for outfit suggestions.")
    print("Got Output")
    return OutfitSuggestions.model_validate_json(interaction.output_text)

def _fetch_missing_image_urls(item_ids: list[str]) -> dict[str, str]:
    """
    Fallback for items whose shortlist projection didn't carry a
    thumbnail_url (e.g. an older document written before that field
    existed). Only hit when a gap is actually detected, and only for
    the specific ids that need it.
    """
    if not item_ids:
        return {}

    collection = _get_wardrobe_collection()
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