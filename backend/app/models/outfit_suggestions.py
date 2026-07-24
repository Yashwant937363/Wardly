from groq import BaseModel

from app.models.outfit import OutfitCombination


class OutfitSuggestions(BaseModel):
    combinations: list[OutfitCombination]