import base64
import os
import time
from google import genai
from pydantic import ValidationError

from app.models.classification import ClassificationResult
from app.models.clothing_analysis import ClothingAnalysis
from app.pipelines.recommendation import OutfitSuggestions

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def classify_image(prompt: str, image_bytes: bytes) -> ClassificationResult :
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
            "thinking_level": "low"
        }
    )

    print("Token Useage(Classify):", interaction.usage)
    print("output: ", interaction.output_text)

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
    
def analyze_image(prompt: str, image_bytes: bytes) -> ClothingAnalysis:
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

def get_recommandations(prompt: str):
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=[{"type": "text", "text": prompt}],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": OutfitSuggestions.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
    )

    if interaction.output_text is None:
        raise RuntimeError("Gemini returned no output for outfit suggestions.")
    try:
        data = OutfitSuggestions.model_validate_json(interaction.output_text)
    except ValidationError as e:
        raise RuntimeError(
            f"Gemini's response didn't match . OutfitSuggestions"
            f"Raw output: {interaction.output_text}"
        ) from e
    return data