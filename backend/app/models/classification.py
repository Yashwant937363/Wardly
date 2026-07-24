from enum import Enum
from typing import List, Optional
from enums.clothing_item import *
from app.models.wardrobe_schema import Category
from pydantic import BaseModel


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

class ImageType(str, Enum):
    PRODUCT_FLAT = "product_flat"
    PRODUCT_MANNEQUIN = "product_mannequin"
    ON_MODEL = "on_model"
    UNUSABLE = "unusable"
 
 
class ClassificationResult(BaseModel):
    image_type: ImageType
    category: GarmentCategory
    wear_category: Category
    reason: str