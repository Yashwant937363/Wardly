import base64
from datetime import datetime
from pathlib import Path
from pydantic import ValidationError
import time

from app.models.classification import ClassificationResult
from app.models.classification import ClassificationResult, GarmentCategory, ImageType
from app.models.clothing_analysis import ClothingAnalysis
from app.services.gemini_client import client
from app.services.rembg_service import extract_with_rembg
from app.services.segformer import extract_with_cloth_parser
from app.services.cloudinary_client import build_thumbnail_url, upload_to_cloudinary
from app.services.mongo_client import save_wardrobe_item
from app.services.fashion_clip import generate_embedding



from app.exceptions import  UnusableImageError

def classify_image_type(image_path: str) -> ClassificationResult:
    """
    Ask Gemini whether the image is a garment-only product shot, a mannequin
    shot, a photo of a person wearing the garment, or unusable.
    """
    from google.genai import types
 
 
    with open(image_path, "rb") as f:
        image_bytes = f.read()
 
    prompt = (
        "Classify this clothing photo.\n\n"
        "image_type — exactly one of:\n"
        "- product_flat: garment alone, laid flat, hung, or on a hanger. No mannequin, no person.\n"
        "- product_mannequin: garment on a mannequin or ghost mannequin. No visible human skin or face.\n"
        "- on_model: a real person is wearing the garment. Skin and/or face visible.\n"
        "- unusable: no single clear garment visible, image too blurry/cropped, or ambiguous.\n\n"
        "category — exactly one of: shirt, tshirt, top, blouse, sweater, hoodie, "
        "pants, jeans, skirt, dress, belt, scarf, shoes, other.\n"
        "Use 'other' for anything that doesn't clearly fit those categories "
        "(watches, jewelry, bags, glasses, hats, etc.).\n"
        "If the photo shows multiple garments, classify only the single most "
        "prominent one."
    )

 
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode('utf-8'),
                "mime_type": "image/jpeg"
            }
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ClassificationResult.model_json_schema()
        },
        generation_config={
            "thinking_level": "minimal"
        }
    )


    if interaction.output_text is None:
        raise RuntimeError(
            "Gemini returned no text content for image classification "
            "(possibly blocked by a safety filter)."
        )

    try:
        data = ClassificationResult.model_validate_json(interaction.output_text)
    except ValidationError as e:
        raise RuntimeError(
            f"Gemini's response didn't match ClothingAnalysis. "
            f"Raw output: {interaction.output_text}"
        ) from e
    return data


_CLOTHING_ANALYSIS_PROMPT = """You are an expert fashion and computer vision assistant.
 
Analyze the provided clothing image and return ONLY a valid JSON object.
 
Rules:
1. Do not include Markdown.
2. Do not wrap the response in ```json```.
3. Do not include explanations.
4. Return only JSON.
5. If a value cannot be determined confidently, use null.
6. Detect only the clothing item shown in the image.
7. Estimate dominant colors as accurately as possible.
8. Percentages of dominant colors should approximately total 100.
 
Return ONLY the JSON."""

def analyze_clothing_image(path: str) -> ClothingAnalysis:
    with open(path, 'rb') as f:
        image_bytes = f.read()

    
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[
            {"type": "text", "text": _CLOTHING_ANALYSIS_PROMPT},
            {
                "type": "image",
                "data": base64.b64encode(image_bytes).decode('utf-8'),
                "mime_type": "image/jpeg"
            }
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ClothingAnalysis.model_json_schema()
        },
        generation_config={
            "thinking_level": "low"
        }
    )

    if interaction.output_text is None:
        raise RuntimeError("Gemini returned no output for clothing analysis.")

    try:
        data = ClothingAnalysis.model_validate_json(interaction.output_text)
    except ValidationError as e:
        raise RuntimeError(
            f"Gemini's response didn't match ClothingAnalysis. "
            f"Raw output: {interaction.output_text}"
        ) from e
    return data


def process_wardrobe_image(
    image_path: str,
    owner_id: str,
    upload_id: str,
    output_path: str | None = None,
) -> dict:
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
          "segmentation": {
            "mask_url": "...",
            "bounding_box": [x0, y0, x1, y1]
          }
        }
    Raises:
        UnusableImageError, UnsupportedCategoryError
    """
    workdir = Path(f"tmp/cutout/{owner_id}/{upload_id}")
    cutout_path = output_path if output_path is not None else str(workdir / "cutout.png")
    mask_path = str(workdir / "mask.png")
    
    classification = classify_image_type(image_path)
 
    if classification.image_type == ImageType.UNUSABLE:
        raise UnusableImageError(classification.reason)
    
 
    if classification.image_type == ImageType.ON_MODEL:
        if classification.category == GarmentCategory.OTHER:
            raise UnusableImageError(
                "This item isn't supported for on-model photos yet. "
                "Please upload a photo of just the item, without a person wearing it."
            )
        cutout_path, mask_path, bounding_box = extract_with_cloth_parser(
            image_path, classification.category.value, cutout_path, mask_path
        )
    else:  # product_flat or product_mannequin — rembg doesn't need a category
        cutout_path, mask_path, bounding_box = extract_with_rembg(
            image_path, cutout_path, mask_path
        )
 
    folder = f"wardrobe/{owner_id}/{upload_id}"
    original_url = upload_to_cloudinary(image_path, folder=folder)
    segmented_url = upload_to_cloudinary(cutout_path, folder=folder)
    mask_url = upload_to_cloudinary(mask_path, folder=folder)
    thumbnail_url = build_thumbnail_url(original_url)
 
    return {
        "image_type": classification.image_type.value,
        "category": classification.category.value,
        "reason": classification.reason,
        "cutout_path": cutout_path,
        "image": {
            "original_url": original_url,
            "thumbnail_url": thumbnail_url,
            "segmented_url": segmented_url,
        },
        "segmentation": {
            "mask_url": mask_url,
            "bounding_box": bounding_box,
        },
    }

def build_wardrobe_document(
    owner_id: str,
    image_result: dict,
    analysis: ClothingAnalysis,
    embedding: list[float],
):
    vision = analysis.model_dump()

    # Add image information
    vision["vision_metadata"]["image"] = image_result["image"]

    # Add segmentation information
    vision["vision_metadata"]["segmentation"] = image_result["segmentation"]

    # Add embedding
    vision["vision_metadata"]["embedding"] = embedding

    return {
        "owner_id": owner_id,
        **vision,
    }

if __name__ == "__main__":
    output_path = "output/shirt.png"
    item_id = "shirt_001"
    owner_id = "user_123"
    category = "shirt"
 
    image_result = process_wardrobe_image(
        image_path="secondtshirt.jpg",
        owner_id=owner_id,
        output_path=output_path,
        upload_id=str(datetime.now())
    )
    metadata_result = analyze_clothing_image(output_path)
    embedding = generate_embedding(output_path)
 
    document = build_wardrobe_document(
        owner_id=owner_id,
        image_result=image_result,
        analysis=metadata_result,
        embedding=embedding,
    )
    saved_id = save_wardrobe_item(document)
    print(f"Saved wardrobe item: {saved_id}")
    # analyze_clothing_image(result["cutout_path"])