import time

from PIL import Image
from pathlib import Path
import numpy as np
from rembg import remove, new_session
from app.exceptions import UnusableImageError

_session = None

def _load_rembg_model():
    global _session
    if _session == None:
        _session = new_session("u2net")
    return _session

def extract_with_rembg(
    image_path: str, output_path: str, mask_output_path: str
) -> tuple[str, str]:
    """
    Generic foreground cutout for garment-only product/mannequin photos
    where the whole foreground IS the garment.
 
    Returns:
        (cutout_path, mask_path, bounding_box)
    """
    t0 = time.time()
    input_image = Image.open(image_path).convert("RGBA")
    session = _load_rembg_model()
    result = remove(input_image, session=session)  # PIL.Image in -> PIL.Image out
 
    if not isinstance(result, Image.Image):
        raise TypeError(f"Expected rembg to return a PIL.Image, got {type(result)}")
 
    alpha = np.array(result)[:, :, 3]
    if alpha.max() == 0:
        raise UnusableImageError("rembg found no foreground subject in this image.")
 
    ys, xs = np.where(alpha > 0)
 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
 
    Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha, mode="L").save(mask_output_path)
    print(f"rembg time: {time.time() - t0:.2f}s")
    return output_path, mask_output_path