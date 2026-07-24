import time
from typing import TYPE_CHECKING, Any

from PIL import Image

 
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
    t0 = time.time()
    model, preprocess = _load_fashion_clip()
 
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  # type: ignore[misc]
 
    with torch.no_grad():
        image_features = model.encode_image(image_tensor, normalize=True)
    print(f"image embeddings: {time.time() - t0:.2f}s")
    return image_features[0].tolist()

_fashion_clip_tokenizer: Any = None


def _load_fashion_clip_tokenizer() -> Any:
    global _fashion_clip_tokenizer
    if _fashion_clip_tokenizer is None:
        import open_clip

        _fashion_clip_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")
    return _fashion_clip_tokenizer


def embed_query(query: str) -> list[float]:
    """
    Embed a text query into the SAME space as the stored image embeddings.
    """
    import torch
    t0 = time.time()
    model, _ = _load_fashion_clip()
    tokenizer = _load_fashion_clip_tokenizer()

    tokens = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(tokens, normalize=True)
    print(f"query embeddings: {time.time() - t0:.2f}s")
    return text_features[0].tolist()