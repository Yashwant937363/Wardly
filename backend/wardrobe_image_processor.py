from dataclasses import dataclass
from datetime import datetime
import time

from google import genai
import base64
import os

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from ultralytics import YOLO

from enum import Enum
from enums.clothing_item import *

import io
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
 
from PIL import Image

from pymongo import MongoClient

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["wardly"]
collection = db["wardrobe_items"]

CATEGORY_TO_SEGFORMER_LABEL: dict[str, list[str]] = {
    "shirt": ["Upper-clothes"],
    "tshirt": ["Upper-clothes"],
    "t-shirt": ["Upper-clothes"],
    "top": ["Upper-clothes"],
    "blouse": ["Upper-clothes"],
    "sweater": ["Upper-clothes"],
    "hoodie": ["Upper-clothes"],
    "pants": ["Pants"],
    "jeans": ["Pants"],
    "trousers": ["Pants"],
    "shorts": ["Pants"],
    "skirt": ["Skirt"],
    "dress": ["Dress"],
    "belt": ["Belt"],
    "scarf": ["Scarf"],
    "shoes": ["Left-shoe", "Right-shoe"],  # combined: a shoe photo needs both
}


class ImageType(str, Enum):
    PRODUCT_FLAT = "product_flat"
    PRODUCT_MANNEQUIN = "product_mannequin"
    ON_MODEL = "on_model"
    UNUSABLE = "unusable"

class GarmentCategory(str, Enum):
    SHIRT = "shirt"
    TSHIRT = "tshirt"
    TOP = "top"
    BLOUSE = "blouse"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    PANTS = "pants"
    JEANS = "jeans"
    SHORTS = "shorts"
    SKIRT = "skirt"
    DRESS = "dress"
    BELT = "belt"
    SCARF = "scarf"
    SHOES = "shoes"
    OTHER = "other"

class DominantColor(BaseModel):
    name: str
    rgb: List[int] = Field(..., min_length=3, max_length=3)
    hsv: List[int] = Field(..., min_length=3, max_length=3)
    hex: str
    percentage: int


class Colors(BaseModel):
    dominant: List[DominantColor]
    primary_color_family: ColorFamily
    secondary_color_family: ColorFamily


class DetectedAttributes(BaseModel):
    category: Category
    subcategory: str
    fit: Fit
    sleeve_length: SleeveLength
    neck_style: NeckStyle 
    pattern: Pattern
    fabric: Fabric
    texture: Texture
    closure: Closure
    neckline: Optional[str] = None
    logo_present: bool
    graphic_present: bool
    hood: bool
    pockets: int


class Confidence(BaseModel):
    category: float
    color: float
    fabric: float
    pattern: float
    fit: float


class VisionMetadata(BaseModel):
    colors: Colors
    detected_attributes: DetectedAttributes
    confidence: Confidence


class ClothingAnalysis(BaseModel):
    vision_metadata: VisionMetadata


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def _cloudinary_configured() -> None:
    import cloudinary
 
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
 
 
def upload_to_cloudinary(file_path: str, folder: str) -> str:
    """Upload a local file to Cloudinary and return its secure_url."""
    import cloudinary.uploader
 
    _cloudinary_configured()
    result = cloudinary.uploader.upload(file_path, folder=folder)
    return result["secure_url"]
 
 
def build_thumbnail_url(original_url: str, width: int = 300) -> str:
    """
    Cloudinary serves resized versions on-the-fly by inserting transformation
    params into the URL path — no separate thumbnail file needs to be stored.
    """
    return original_url.replace(
        "/upload/", f"/upload/w_{width},c_fit,q_auto,f_auto/"
    )


 
 
class ImageTypeOutput(BaseModel):
    """Schema Gemini must fill in for structured output."""
 
    image_type: ImageType
    category: GarmentCategory
    reason: str
 
 
@dataclass
class ClassificationResult:
    image_type: ImageType
    category: GarmentCategory
    reason: str
 
 
class UnusableImageError(Exception):
    """Raised when Gemini flags the image as unusable for cataloging."""
 
 
class UnsupportedCategoryError(Exception):
    """Raised when the given category has no cloth-parser mapping."""
 
 
# ----------------------------------------------------------------------
# Step 1: Gemini classification
# ----------------------------------------------------------------------
 
def classify_image_type(image_path: str) -> ClassificationResult:
    """
    Ask Gemini whether the image is a garment-only product shot, a mannequin
    shot, a photo of a person wearing the garment, or unusable.
    """
    from google import genai
    from google.genai import types
 
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
 
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
 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ImageTypeOutput,
        ),
    )
 
    raw_text = response.text
    if raw_text is None:
        raise RuntimeError(
            "Gemini returned no text content for image classification "
            "(possibly blocked by a safety filter)."
        )
 
    data = ImageTypeOutput.model_validate_json(raw_text)
    return ClassificationResult(
        image_type=data.image_type, category=data.category, reason=data.reason
    )
 
 
# ----------------------------------------------------------------------
# Step 2a: on_model images -> cloth parser (Segformer, garment-on-body)
# ----------------------------------------------------------------------
from typing import TYPE_CHECKING, Any
 
if TYPE_CHECKING:
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
 
_segformer_processor: "SegformerImageProcessor | None" = None
_segformer_model: "AutoModelForSemanticSegmentation | None" = None
 
def _load_segformer() -> tuple[Any, Any]:
    global _segformer_processor, _segformer_model
    if _segformer_model is None:
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
 
        _segformer_processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        _segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
    return _segformer_processor, _segformer_model

from typing import TYPE_CHECKING, Any

 
if TYPE_CHECKING:
    from torch.nn import Module
 
_fashion_clip_model: "Module | None" = None
_fashion_clip_preprocess: Any = None
 
 
def _load_fashion_clip() -> tuple[Any, Any]:
    """
    Load Marqo-FashionCLIP directly through open_clip, bypassing the
    transformers `trust_remote_code` wrapper. The wrapper's custom __init__
    calls open_clip.create_model(...) and .to(device) itself, which is
    incompatible with transformers/accelerate's meta-device lazy init
    (causes "Cannot copy out of meta tensor; no data!" depending on
    installed accelerate/transformers versions). Loading via open_clip
    directly avoids that machinery entirely.
    """
    global _fashion_clip_model, _fashion_clip_preprocess
    if _fashion_clip_model is None:
        import open_clip
 
        _fashion_clip_model, _, _fashion_clip_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionCLIP"
        )
        _fashion_clip_model.eval()
    return _fashion_clip_model, _fashion_clip_preprocess
 
 
def generate_embedding(image_path: str) -> list[float]:
    """
    Generate a 512-dim, unit-normalized embedding for a garment cutout,
    for similarity search / outfit matching via MongoDB Atlas Vector Search.
    """
    import torch
 
    model, preprocess = _load_fashion_clip()
 
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  # type: ignore[misc]
 
    with torch.no_grad():
        image_features = model.encode_image(image_tensor, normalize=True)
 
    return image_features[0].tolist()

_mongo_client: Any = None

def _get_wardrobe_collection():
    """
    Lazily create a single MongoClient for the process lifetime (MongoClient
    is thread-safe and pools connections internally, so it should NOT be
    recreated per call/request).
    """
    global _mongo_client
    from pymongo import MongoClient
 
    if _mongo_client is None:
        _mongo_client = MongoClient(os.environ["MONGODB_URI"])
 
    db_name = os.environ.get("MONGODB_DB", "wardly")
    return _mongo_client[db_name]["wardrobe_items"]
 
def extract_with_cloth_parser(
    image_path: str, category: str, output_path: str, mask_output_path: str
) -> tuple[str, str, list[int]]:
    """
    Isolate a single garment from an on-model photo using Segformer human
    parsing, and save it as a transparent PNG cutout.
 
    Returns:
        (cutout_path, mask_path, bounding_box)
    """
    import numpy as np
    import torch
    import torch.nn as nn
 
    label_names = CATEGORY_TO_SEGFORMER_LABEL.get(category.lower())
    if label_names is None:
        raise UnsupportedCategoryError(
            f"'{category}' is not supported by the cloth parser. "
            f"Supported: {sorted(CATEGORY_TO_SEGFORMER_LABEL.keys())}"
        )
 
    processor, model = _load_segformer()
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    target_ids = [label2id[name] for name in label_names]
 
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
 
    with torch.no_grad():
        outputs = model(**inputs)
 
    logits = outputs.logits.cpu()
    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled.argmax(dim=1)[0].numpy()
 
    mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255
    if mask.sum() == 0:
        raise UnusableImageError(
            f"Cloth parser found no {label_names} region in this image for category '{category}'."
        )
 
    rgba = image.convert("RGBA")
    rgba_array = np.array(rgba)
    rgba_array[:, :, 3] = mask  # alpha channel = garment mask
 
    cropped, bounding_box = _crop_to_mask(rgba_array, mask)
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
 
    Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_output_path)  # plain black/white mask
 
    return output_path, mask_output_path, bounding_box
 
 
def _crop_to_mask(rgba_array, mask):
    import numpy as np
 
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cropped_array = rgba_array[y0 : y1 + 1, x0 : x1 + 1]
    bounding_box = [x0, y0, x1, y1]  # relative to the ORIGINAL uploaded image
    return Image.fromarray(cropped_array, mode="RGBA"), bounding_box
 
 
# ----------------------------------------------------------------------
# Step 2b: product_flat / product_mannequin -> rembg (generic cutout)
# ----------------------------------------------------------------------
 
def extract_with_rembg(
    image_path: str, output_path: str, mask_output_path: str
) -> tuple[str, str, list[int]]:
    """
    Generic foreground cutout for garment-only product/mannequin photos
    where the whole foreground IS the garment.
 
    Returns:
        (cutout_path, mask_path, bounding_box)
    """
    import numpy as np
    from rembg import remove
 
    input_image = Image.open(image_path).convert("RGBA")
    result = remove(input_image)  # PIL.Image in -> PIL.Image out
 
    if not isinstance(result, Image.Image):
        raise TypeError(f"Expected rembg to return a PIL.Image, got {type(result)}")
 
    alpha = np.array(result)[:, :, 3]
    if alpha.max() == 0:
        raise UnusableImageError("rembg found no foreground subject in this image.")
 
    ys, xs = np.where(alpha > 0)
    bounding_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
 
    Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha, mode="L").save(mask_output_path)
 
    return output_path, mask_output_path, bounding_box
 
 
 
# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
 
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
    workdir = Path(f"tmp/{owner_id}/{upload_id}")
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

def get_data(path: str) -> ClothingAnalysis:
    with open(path, 'rb') as f:
        image_bytes = f.read()

    client = genai.Client(api_key=GEMINI_API_KEY)
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

def save_wardrobe_item(document: dict):
    collection.insert_one(document)

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
    metadata_result = get_data(output_path)
    embedding = generate_embedding(output_path)
 
    document = build_wardrobe_document(
        owner_id=owner_id,
        image_result=image_result,
        analysis=metadata_result,
        embedding=embedding,
    )
    saved_id = save_wardrobe_item(document)
    print(f"Saved wardrobe item: {saved_id}")
    # get_data(result["cutout_path"])
    
