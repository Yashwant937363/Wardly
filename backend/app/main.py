"""
FastAPI backend for wardrobe item upload + outfit suggestions.

Run with:
    uv run --env-file .env uvicorn main:app --reload

Requires:
    uv add fastapi "uvicorn[standard]" python-multipart
"""


import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.services.mongo_client import _mongo_safe, collection
from app.api.items import router as item_router
from app.api.suggestions import router as suggestions_router
from app.services.fashion_clip import _load_fashion_clip_tokenizer, _load_fashion_clip
from app.services.rembg_service import _load_rembg_model
from app.services.cloudinary_client import _cloudinary_configured

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    model, _ = _load_fashion_clip()
    if model != None:
        print("Fashion Clip Model Loaded")
    tokenizer = _load_fashion_clip_tokenizer()
    if tokenizer != None:
        print("Fasion Tokenizer Model Loaded")
    session = _load_rembg_model()
    if session != None:
        print("RemBG Model Loaded")
    yield
    _cloudinary_configured()
    # Run shutdown functions
    print("Server shutting down...")
    

app = FastAPI(title="Wardly API", lifespan=lifespan)


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

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")