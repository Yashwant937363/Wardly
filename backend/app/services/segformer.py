import time
from typing import TYPE_CHECKING, Any

from PIL import Image
from pathlib import Path

from app.exceptions import UnsupportedCategoryError, UnusableImageError

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


def _crop_to_mask(rgba_array, mask):
    import numpy as np
 
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cropped_array = rgba_array[y0 : y1 + 1, x0 : x1 + 1]
    return Image.fromarray(cropped_array, mode="RGBA")

def extract_with_cloth_parser(
    image_path: str, category: str, output_path: str, mask_output_path: str
) -> tuple[str, str]:
    """
    Isolate a single garment from an on-model photo using Segformer human
    parsing, and save it as a transparent PNG cutout.
 
    Returns:
        (cutout_path, mask_path, bounding_box)
    """
    import numpy as np
    import torch
    import torch.nn as nn
    t0 = time.time()
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
 
    cropped= _crop_to_mask(rgba_array, mask)
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path)
 
    Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_output_path)  # plain black/white mask
    print(f"cloth parser: {time.time() - t0:.2f}s")
    return output_path, mask_output_path