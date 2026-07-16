import uuid
import asyncio
from pathlib import Path
from google.genai.errors import ServerError
from fastapi import APIRouter,FastAPI, File, Form, HTTPException, UploadFile

from app.exceptions import UnsupportedCategoryError, UnusableImageError

from app.services.fashion_clip import generate_embedding
from app.services.mongo_client import collection,_mongo_safe, save_wardrobe_item
from app.pipelines.upload_pipeline import analyze_clothing_image, build_wardrobe_document, process_wardrobe_image

UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

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
        asyncio.to_thread(analyze_clothing_image, output_path),
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

@router.post("/api/items")
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