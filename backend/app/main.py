"""
FastAPI backend for wardrobe item upload + outfit suggestions.

Run with:
    uv run --env-file .env uvicorn main:app --reload

Requires:
    uv add fastapi "uvicorn[standard]" python-multipart
"""


import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services.mongo_client import _mongo_safe, collection
from app.api.items import router as item_router
from app.api.suggestions import router as suggestions_router






app = FastAPI(title="Wardly API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_all_wardrobe_items():
    return list(collection.find())

@app.get("/api/items")
async def get_all_items():
    try:
        documents = await asyncio.to_thread(get_all_wardrobe_items)
        return [_mongo_safe(doc) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(item_router, prefix="/api")
app.include_router(suggestions_router, prefix="/api")

app.mount("/", StaticFiles(directory="static", html=True), name="static")