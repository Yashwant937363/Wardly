

from typing import Type, TypeVar

from app.prompts import BASE_ANALYSIS_PROMPT, get_metadata_prompt


import asyncio
import base64
from datetime import datetime
import io
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, TypeAdapter, ValidationError
import time

from app.models.classification import ClassificationResult
from app.models.classification import ClassificationResult, GarmentCategory, ImageType
from app.models.clothing_analysis import ClothingAnalysis
from app.models.wardrobe_schema import Category, Colors, ConfidenceScores, ImageInfo, WardrobeItem, WardrobeItemBase, get_metadata_class
from app.services.groq_client import analyze_image, classify_image
from app.services.rembg_service import extract_with_rembg
from app.services.segformer import extract_with_cloth_parser
from app.services.cloudinary_client import build_thumbnail_url, upload_to_cloudinary, get_cloudinary_folder_path
from app.services.mongo_client import save_wardrobe_item
from app.services.fashion_clip import generate_embedding



from app.exceptions import  UnusableImageError



def classify_image_type(image_path: str) -> ClassificationResult:
    """
    Ask Gemini whether the image is a garment-only product shot, a mannequin
    shot, a photo of a person wearing the garment, or unusable.
    """
    from google.genai import types
    t0 = time.time()

    image = Image.open(image_path)
    image.thumbnail((500,500))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", quality=75, optimize=True)


    image_bytes = buffer.getvalue()
    
    prompt = """
    # TASK
    Expert fashion image classifier. Analyze the image and output EXACTLY ONE valid JSON object, no markdown, no code fences, no extra text.

    # OUTPUT FORMAT
    {
    "image_type": "<product_flat|product_mannequin|on_model|unusable>",
    "wear_category": "<Topwear|Bottomwear|Dresswear|Drapewear|Outerwear|Innerwear|Footwear|Hosiery|Headwear|Eyewear|Earwear|Neckwear|Wristwear|Anklewear|Bags|Waistwear>",
    "category": "<shirt|tshirt|top|blouse|sweater|hoodie|pants|jeans|shorts|skirt|dress|belt|scarf|shoes|other>",
    "reason": "<short explanation, under 25 words>"
    }

    # image_type
    - product_flat: garment alone, flat/hanging/on hanger, no mannequin, no person.
    - product_mannequin: garment on a mannequin/ghost mannequin, no visible skin or face.
    - on_model: worn by a real person (any visible skin, face, hands, limbs counts).
    - unusable: blurry, cropped, no clothing/accessory, multiple equally prominent items, or otherwise ambiguous.

    # wear_category
    Choose the broad body-zone group the item belongs to:
    - Topwear: shirts, t-shirts, blouses, tops, sweaters, hoodies
    - Bottomwear: pants, jeans, shorts, skirts, trousers
    - Dresswear: dresses, gowns, jumpsuits, one-piece garments
    - Drapewear: sarees, dupattas, shawls, wraps worn draped over the body
    - Outerwear: jackets, coats, blazers, cardigans worn over other clothing
    - Innerwear: underwear, bras, undershirts
    - Footwear: shoes, sneakers, boots, sandals, heels
    - Hosiery: socks, stockings, tights
    - Headwear: hats, caps, beanies
    - Eyewear: glasses, sunglasses
    - Earwear: earrings, ear cuffs
    - Neckwear: necklaces, scarves, ties, chokers
    - Wristwear: watches, bracelets, wristbands
    - Anklewear: anklets, ankle bracelets
    - Bags: handbags, backpacks, totes, clutches
    - Waistwear: belts, waist chains

    # category
    Identify the specific item type. Use one of: shirt, tshirt, top, blouse, sweater, hoodie, pants, jeans, shorts, skirt, dress, belt, scarf, shoes, other (use "other" if none fit).

    IMPORTANT: A watch is never a belt. A bag is never footwear. Match "category" and "wear_category" consistently — e.g. a watch must be category="watch" AND wear_category="Wristwear", never wear_category="Waistwear".

    # RULES
    - Classify only the single most prominent item in the image; ignore background.
    - Base decisions only on visible evidence — don't invent details.
    - wear_category and category must describe the SAME item and must be logically consistent with each other.

    Return ONLY the JSON object.
    """
    data = classify_image(prompt=prompt, image_bytes=image_bytes)
    
    print(f"classify: {time.time() - t0:.2f}s")
    return data


class WardrobeItemAnalysis(BaseModel):
    name: str
    subType: str | None = None

    colors: Colors
    material: str

    season: list[str] = []
    occasion: list[str] = []

    confidence: ConfidenceScores | None = None

T = TypeVar("T", bound=BaseModel)
adapter = TypeAdapter(WardrobeItem)
async def analyze_clothing_image(path: str, category: Category) -> WardrobeItem:
    with open(path, 'rb') as f:
        image_bytes = f.read()

    t0 = time.time()

    metadata_prompt = get_metadata_prompt(category=category)
    metadata_schema = get_metadata_class(category) if metadata_prompt != "" else None

    if metadata_prompt != "" and metadata_schema:
        common_data, meta_data = await asyncio.gather(
            asyncio.to_thread(analyze_image, prompt=BASE_ANALYSIS_PROMPT, image_bytes=image_bytes, schema=WardrobeItemAnalysis),
            asyncio.to_thread(analyze_image, prompt=metadata_prompt, image_bytes=image_bytes, schema=metadata_schema),
        )
        metadata_dict = {"metadata": meta_data.model_dump()}
    else:
        common_data = await asyncio.to_thread(
            analyze_image, prompt=BASE_ANALYSIS_PROMPT, image_bytes=image_bytes, schema=WardrobeItemAnalysis
        )
        metadata_dict = {}

    combined = {
        **common_data.model_dump(),
        "category": category,
        **metadata_dict,

        # Placeholder values
        "image": {
            "original_url": "",
            "thumbnail_url": "",
            "segmented_url": "",
        },
        "embedding": [],
    }

    data = adapter.validate_python(combined)

    print(f"analyze: {time.time() - t0:.2f}s")
    return data

class Process_Wardrobe_Image_Result(BaseModel):
    category: Category
    reason: str
    cutout_path: str
    image: ImageInfo

async def process_wardrobe_image(
    image_path: str,
    owner_id: str,
    upload_id: str,
    output_path: str | None = None,
) -> Process_Wardrobe_Image_Result:
    """
    Full routing + upload pipeline for a single uploaded wardrobe item photo.
 
    Category is no longer a caller-supplied parameter — Gemini detects it
    in the same classification call as image_type, so the frontend doesn't
    need a category dropdown anymore.
 
    Args:
        upload_id: a correlation ID used only to namespace local temp files
            and Cloudinary folders for THIS upload. It's not the item's
            database identity — that's MongoDB's own _id, assigned when the
            resulting document is inserted.
        output_path: where to save the local cutout PNG. If not given,
            defaults to tmp/{owner_id}/{upload_id}/cutout.png. This path is
            also returned as "cutout_path" so it can be fed straight into
            a follow-up Gemini call for fashion_metadata extraction.
 
    Returns a dict matching the vision_metadata.image / .segmentation shape,
    plus the detected category:
        {
          "image_type": "...",
          "category": "...",
          "reason": "...",
          "cutout_path": "...",   # local path, NOT the same as segmented_url
          "image": {
            "original_url": "...",
            "thumbnail_url": "...",
            "segmented_url": "..."
          },
        }
    Raises:
        UnusableImageError, UnsupportedCategoryError
    """
    folder: str = get_cloudinary_folder_path(owner_id=owner_id, upload_id=upload_id)
    workdir = Path(f"tmp/cutout/{owner_id}/{upload_id}")
    cutout_path = output_path if output_path is not None else str(workdir / "cutout.png")
    mask_path = str(workdir / "mask.png")
    
    classification, original_url = await asyncio.gather(
        asyncio.to_thread(classify_image_type, image_path),
        asyncio.to_thread(upload_to_cloudinary, image_path,folder=folder)
    )
 
    if classification.image_type == ImageType.UNUSABLE:
        raise UnusableImageError(classification.reason)
    
 
    if classification.image_type == ImageType.ON_MODEL:
        if classification.category == GarmentCategory.OTHER:
            raise UnusableImageError(
                "This item isn't supported for on-model photos yet. "
                "Please upload a photo of just the item, without a person wearing it."
            )
        cutout_path, mask_path = extract_with_cloth_parser(
            image_path, classification.category.value, cutout_path, mask_path
        )
    else:  # product_flat or product_mannequin — rembg doesn't need a category
        cutout_path, mask_path = extract_with_rembg(
            image_path, cutout_path, mask_path
        )
 
    t0 = time.time()
    thumbnail_url = build_thumbnail_url(original_url)
    print(f"cloundnary time: {time.time() - t0:.2f}s")
    result = Process_Wardrobe_Image_Result(
        category=classification.wear_category,
        reason=classification.reason,
        cutout_path=cutout_path,
        image=ImageInfo(
            original_url= original_url,
            thumbnail_url = thumbnail_url,
            segmented_url = cutout_path,
        )
    )
    # {
    #     "category": classification.wear_category.value,
    #     "reason": classification.reason,
    #     "cutout_path": cutout_path,
    #     "image": {
    #         "original_url": original_url,
    #         "thumbnail_url": thumbnail_url,
    #         "segmented_url": cutout_path,
    #     },
    # }
    return result

def build_wardrobe_document(
    owner_id: str,
    image_result: ImageInfo,
    analysis: WardrobeItem,
    embedding: list[float],
):
    data = analysis

    # Add image information
    data.image = image_result

    # Add segmentation information

    # Add embedding
    data.embedding = embedding
    data = data.model_dump()
    return {
        "owner_id": owner_id,
        **data,
    }

if __name__ == "__main__":
    output_path = "output/shirt.png"
    item_id = "shirt_001"
    owner_id = "user_123"
 
    print("Script Running ")

    image_result = asyncio.run(
        process_wardrobe_image(
            image_path="/home/yashwant/Desktop/Wardrobe/tshirts/tshirt-7.jpg",
            owner_id=owner_id,
            output_path=output_path,
            upload_id=str(datetime.now())
        )
    )
    print(image_result)
    # metadata_result = analyze_clothing_image(output_path)
    embedding = generate_embedding(output_path)
 
    # document = build_wardrobe_document(
    #     owner_id=owner_id,
    #     image_result=image_result,
    #     analysis=metadata_result,
    #     embedding=embedding,
    # )
    # saved_id = save_wardrobe_item(document)
    # print(f"Saved wardrobe item: {saved_id}")
    