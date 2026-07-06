"""
FastAPI backend for wardrobe item upload + outfit suggestions.

Run with:
    uv run --env-file .env uvicorn api:app --reload

Requires:
    uv add fastapi "uvicorn[standard]" python-multipart
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any

from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google.genai.errors import ServerError

from ai_agent import OutfitSuggestions, get_outfit_suggestions
from wardrobe_image_processor import (
    UnsupportedCategoryError,
    UnusableImageError,
    build_wardrobe_document,
    generate_embedding,
    get_data,
    process_wardrobe_image,
    save_wardrobe_item,
)


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

app = FastAPI(title="Wardly API")

# Tighten allow_origins to your actual frontend domain before shipping this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def process_and_store_item(
    image_path: str, owner_id: str, upload_id: str, output_path: str
) -> dict:
    """
    process_wardrobe_image must complete first — get_data and
    generate_embedding both consume the cutout it produces. Once the cutout
    exists, those two are independent (one's a Gemini network call, the
    other's local model inference), so they run concurrently.

    Category is no longer passed in — Gemini detects it inside
    process_wardrobe_image and it comes back on routing_result["category"].
    """
    routing_result = await asyncio.to_thread(
        process_wardrobe_image,
        image_path=image_path,
        owner_id=owner_id,
        upload_id=upload_id,
        output_path=output_path,
    )

    analysis, embedding = await asyncio.gather(
        asyncio.to_thread(get_data, output_path),
        asyncio.to_thread(generate_embedding, output_path),
    )

    document = build_wardrobe_document(
        owner_id=owner_id,
        image_result=routing_result,
        analysis=analysis,
        embedding=embedding,
    )
    await asyncio.to_thread(save_wardrobe_item, document)
    return document


@app.post("/api/items")
async def upload_item(
    file: UploadFile = File(...),
    owner_id: str = Form("user_123"),  # replace with real auth once you have it
):
    upload_id = uuid.uuid4().hex[:8]
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    output_path = str(OUTPUT_DIR / f"{upload_id}.png")

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    try:
        document = await process_and_store_item(
            image_path=str(upload_path),
            owner_id=owner_id,
            upload_id=upload_id,
            output_path=output_path,
        )
    except UnusableImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except UnsupportedCategoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServerError:
        raise HTTPException(
            status_code=503,
            detail="Image analysis service is temporarily unavailable. Please try again in a moment.",
        )

    return _mongo_safe(document)


# ----------------------------------------------------------------------
# Outfit suggestions
# ----------------------------------------------------------------------

class OutfitQueryRequest(BaseModel):
    query: str
    owner_id: str = "user_123"  # replace with real auth once you have it


@app.post("/api/outfit-suggestions", response_model=OutfitSuggestions)
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


# Serve the upload page — must be mounted AFTER the routes above so
# "/api/items" and "/api/outfit-suggestions" are matched first and aren't
# swallowed by this catch-all.
app.mount("/", StaticFiles(directory="static", html=True), name="static")