from pydantic import BaseModel, Field
from enums.clothing_item import *
from typing import List, Optional

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