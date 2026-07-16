
from pydantic import BaseModel


class OutfitItemRef(BaseModel):
    item_id: str
    category: str
    reason: str
    image_url: str | None = None


class OutfitCombination(BaseModel):
    title: str
    items: list[OutfitItemRef]
    styling_notes: str