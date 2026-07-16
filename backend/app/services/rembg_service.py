from PIL import Image
from pathlib import Path

from app.exceptions import UnusableImageError


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