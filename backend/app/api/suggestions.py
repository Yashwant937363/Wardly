import asyncio

from fastapi import APIRouter, HTTPException
from google.genai.errors import ServerError
from pydantic import BaseModel
from app.pipelines.recommendation import OutfitSuggestions, get_outfit_suggestions

router = APIRouter(prefix="/outfit-suggestions")


class OutfitQueryRequest(BaseModel):
    query: str
    owner_id: str = "user_123"  # replace with real auth once you have it


@router.post("/", response_model=OutfitSuggestions)
async def outfit_suggestions(payload: OutfitQueryRequest):
    query = payload.query.strip()
    print("Query: ",query)
    print("Owner Id: ", payload.owner_id)
    if not query:
        raise HTTPException(status_code=400, detail="Query can't be empty.")

    try:
        # get_outfit_suggestions is blocking (FashionCLIP inference, a Mongo
        # vector search, and a Gemini call), so it runs off the event loop
        # the same way the upload pipeline does above.
        suggestions = await asyncio.to_thread(
            get_outfit_suggestions, query, payload.owner_id
        )
    except ValueError as e:
        # get_outfit_suggestions raises ValueError both when the closet is
        # empty and when Gemini hallucinates an item_id outside the
        # shortlist — the former is a client-fixable 404, the latter is an
        # upstream generation issue, so it gets a 502 instead.
        message = str(e)
        if "No wardrobe items found" in message:
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=502, detail=message)
    except ServerError:
        raise HTTPException(
            status_code=503,
            detail="Styling service is temporarily unavailable. Please try again in a moment.",
        )

    return suggestions