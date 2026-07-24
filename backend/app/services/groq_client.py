import base64
import os
from typing import Type, TypeVar

import groq
from pydantic import BaseModel, ValidationError

from app.models.classification import ClassificationResult
from app.models.clothing_analysis import ClothingAnalysis
from app.models.outfit_suggestions import OutfitSuggestions

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=GROQ_API_KEY)

def classify_image(prompt: str, image_bytes: bytes) -> ClassificationResult :
    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                        },
                    }
                ],
            },
        ],
        temperature=0.3,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="none",
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    print("Token Useage(Classify):", completion.usage)
    print("output: ", completion.choices[0].message.content)

    if completion.choices[0].message.content is None:
        raise RuntimeError(
            "Gemini returned no text content for image classification "
            "(possibly blocked by a safety filter)."
        )

    try:
        data = ClassificationResult.model_validate_json(completion.choices[0].message.content)
    except ValidationError as e:
        raise RuntimeError(
            f"Gemini's response didn't match ClothingAnalysis. "
            f"Raw output: {completion.choices[0].message.content}"
        ) from e
    return data

T = TypeVar("T", bound=BaseModel)

def analyze_image(prompt: str, image_bytes: bytes, schema: Type[T]) -> T:
    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                        },
                    }
                ],
            },
        ],
        temperature=0.3,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="none",
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    print("Token Useage(Analyze):", completion.usage)

    if completion.choices[0].message.content is None:
            raise RuntimeError(
                "Groq returned no text content for image analyze"
                "(possibly blocked by a safety filter)."
            )
    
    try:
        data = schema.model_validate_json(completion.choices[0].message.content)
        
    except ValidationError as e:
        raise RuntimeError(
            f"Groq's response didn't match ClothingAnalysis."
            f"Raw output: {completion.choices[0].message.content}"
        ) from e
    return data

def get_recommandations(system_prompt: str, user_prompt: str) -> OutfitSuggestions:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        max_completion_tokens=2048,
        top_p=0.95,
        reasoning_effort="medium",
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "outfit_suggestion",
                "schema": OutfitSuggestions.model_json_schema()
            }
        },
        stop=None,
    )
    print("Token Usage (Analyze):", completion.usage)

    if completion.choices[0].message.content is None:
        raise RuntimeError("Model returned no output for outfit suggestions.")

    try:
        data = OutfitSuggestions.model_validate_json(completion.choices[0].message.content)
    except ValidationError as e:
        raise RuntimeError(
            f"Model's response didn't match OutfitSuggestions schema. "
            f"Raw output: {completion.choices[0].message.content}"
        ) from e

    return data