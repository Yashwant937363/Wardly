"""
Two-stage clothing analysis prompts.

Stage 1 (BASE_ANALYSIS_PROMPT): run on every image, regardless of category.
Produces exactly the fields on `WardrobeItemBase`: category, subType, colors,
material, season, occasion, and the *shared* confidence keys (overall,
category, color, fabric). Fit/pattern confidence are scored in Stage 2,
since they only make sense once the metadata shape is known.

Stage 2 (CATEGORY_METADATA_PROMPTS[category]): run only after Stage 1 has
told you the `category`. Each prompt asks for exactly the fields on that
category's `*Metadata` Pydantic class — nothing more, nothing less — plus
confidence scores for those specific fields (merged into ConfidenceScores'
extra="allow" bucket, e.g. {"heelHeight": 0.88}).

Two categories (Innerwear, Headwear) have empty metadata classes in the
schema, so there is no Stage-2 prompt for them — Stage 1 alone is enough.

DESIGN NOTE — no closed enums:
Every field below is documented, not enumerated. Each description explains
what the field means and gives a few illustrative examples ("e.g. ..."),
but the model is free to write whatever value actually matches the image —
including something not listed — rather than being forced into the closest
wrong bucket from a fixed list. The trade-off is enforced with instructions
instead of a hard whitelist: keep values short, concrete, and drawn only
from what's visibly true of the item, not invented or generic filler.

Usage:

    base_result = call_llm(BASE_ANALYSIS_PROMPT, image)
    category = base_result["category"]  # e.g. "Footwear"
    if category in CATEGORY_METADATA_PROMPTS:
        meta_result = call_llm(CATEGORY_METADATA_PROMPTS[category], image,
                                context={"category": category, "subType": base_result["subType"]})
    else:
        meta_result = {}
"""

# ==========================================================================
# STAGE 1 — COMMON / BASE PROMPT (all categories)
# ==========================================================================

from app.models.wardrobe_schema import Category


BASE_ANALYSIS_PROMPT:str = """You are an expert fashion and computer vision assistant.

Analyze the clothing/accessory image and return ONLY a valid JSON object matching the schema below. No markdown, no ```json``` fences, no explanations — JSON only.

# SCHEMA

{
  "name":"<string>",
  "subType": "<string|null>",
  "colors": {
    "dominant": [
      {"name": "<string>", "rgb": [R,G,B], "percentage": <int>}
    ],
    "primary_color_family": "<string>",
    "secondary_color_family": "<string|null>"
  },
  "material": "<string>",
  "season": ["<string>", "..."],
  "occasion": ["<string>", "..."],
  "confidence": {
    "overall": <float 0-1>,
    "category": <float 0-1>,
    "color": <float 0-1>,
    "fabric": <float 0-1>
  }
}

# FIELD GUIDANCE

Every field is free text — write exactly what you see, in your own words. The notes below explain the intent of each field and give a few illustrative examples; they are not a closed list, so if the item doesn't match any example, describe it accurately instead of forcing it into one.

- name: a short, descriptive, human-readable label for this specific item, the way it would appear as a title in a wardrobe app — typically color + distinguishing detail + subType, e.g. "Navy Blue Checked Trousers", "Black Slim-Fit Dress Shirt", "Rust Orange Embroidered Kurti", "Silver-Tone Leather Strap Watch". Keep it to 3-6 words. Do not just repeat subType alone — name should be specific enough to tell this item apart from a similar one of the same subType.
- subType: the specific type of item within that category, e.g. "T-Shirt", "Polo Shirt", "Blouse", "Kurti" for Topwear; "Jeans", "Trousers", "Salwar" for Bottomwear; "Saree", "Lehenga", "Dupatta" for Drapewear; "Jacket", "Blazer", "Sherwani" for Outerwear; "Sneakers", "Loafers", "Sandals" for Footwear; "Belt" for Waistwear; "Necklace", "Scarf", "Tie" for Neckwear; and so on. Name the item the way a shopper would, in 1-3 words. Use null only if you genuinely can't tell what the item is.
- colors.dominant: one entry per visually significant color patch on the item. "name" should be a natural color name a person would use (e.g. "Navy Blue", "Rust Orange", "Off-white"), not a hex code. "percentage" is your best estimate of how much of the visible item that color covers; percentages across the list should sum to roughly 100.
- colors.primary_color_family: the single broad color family the item would be filed under in a closet (e.g. "Blue", "Black", "Multicolor", "Neutral").
- colors.secondary_color_family: a second broad color family if the item is genuinely two-toned or multi-colored; otherwise null.
- material: the fabric or material the item is made from, as you'd describe it in a product listing (e.g. "Cotton", "Denim", "Sterling Silver", "Faux Leather"). If you can't tell from the image, write "Unknown".
- season: which season(s) this item would realistically be worn in, as a short list (e.g. ["Winter"], ["Summer", "Monsoon"], ["All-Season"]). Base this on the material weight and cut, not guesswork about the photo's setting.
- occasion: which occasion(s) this item suits, as a short list (e.g. ["Casual"], ["Formal", "Work"], ["Party"], ["Ethnic", "Festive"]). Only list occasions you can actually justify from the item's style — don't pad the list.
- confidence.overall / category / color / fabric: your genuine certainty for each judgment, as a float between 0 and 1. Lower the score honestly when the image is ambiguous, cropped, poorly lit, or the item is partially obscured — don't default to a high number.

# RULES

1. Every value must be something you can justify from what's actually visible in the image — never invent details, and never fall back to a generic placeholder just to fill the field.
2. If a field genuinely can't be determined, use null rather than guessing.
3. Detect only the single main item in the image.
4. `season` and `occasion` are arrays — an item can belong to more than one, but don't over-list.
5. Confidence scores are floats between 0 and 1.
6. Output must be valid, parseable JSON — no trailing commas, no comments.

Return ONLY the JSON object."""


# ==========================================================================
# STAGE 2 — CATEGORY-SPECIFIC METADATA PROMPTS
# ==========================================================================
# Each prompt below mirrors one *Metadata Pydantic class field-for-field.
# The {category} / {subtype} placeholders should be filled with Stage 1's
# output before sending, so the model isn't re-guessing what it already
# determined.

_PROMPT_HEADER = """You are an expert fashion and computer vision assistant.

This image has already been classified as category="{category}".
Analyze ONLY the attributes below for this specific item and return ONLY a valid JSON object matching the schema. No markdown, no ```json``` fences, no explanations — JSON only.

# SCHEMA
{schema}

# FIELD GUIDANCE
{guidance}

# RULES
1. Every value is free text — describe what you actually see; don't force it into an example below if it doesn't fit, and don't invent detail you can't justify from the image.
2. If a field genuinely can't be determined, use null.
3. Confidence scores are floats between 0 and 1 — lower them honestly for ambiguous or partially-visible attributes.
4. Output must be valid, parseable JSON — no trailing commas, no comments.

Return ONLY the JSON object."""


def _build(schema_lines: str, guidance_lines: str) -> str:
    return _PROMPT_HEADER.replace("{schema}", schema_lines).replace("{guidance}", guidance_lines)


CATEGORY_METADATA_PROMPTS: dict[Category, str] = {

    # ---- Topwear -> TopwearMetadata(sleeveType, neckline, fit, pattern, length) ----
    Category.TOPWEAR: _build(
        schema_lines="""{
  "sleeveType": "<string>",
  "neckline": "<string>",
  "fit": "<string>",
  "pattern": "<string>",
  "length": "<string>"
}""",
        guidance_lines="""- sleeveType: how the sleeves are cut, e.g. "Full", "Half", "Sleeveless", "3-quarter", "Cap Sleeve". Describe what's visible, not a guess.
- neckline: the collar/neckline shape, e.g. "Round", "V-neck", "Collar", "Boat neck", "Halter Neck", "Turtleneck", "Mandarin Collar".
- fit: how the garment sits on the body, e.g. "Slim", "Loose", "Regular", "Oversized".
- pattern: the surface design, e.g. "Solid", "Striped", "Printed", "Checked", "Floral", "Graphic", "Polka Dot".
- length: how far down the torso it extends, e.g. "Regular", "Crop", "Longline"."""
    ),

    # ---- Bottomwear -> BottomwearMetadata(fit, length, waistRise, closure[]) ----
    Category.BOTTOMWEAR: _build(
        schema_lines="""{
  "fit": "<string>",
  "length": "<string>",
  "waistRise": "<string>",
  "closure": ["<string>", "..."]
}""",
        guidance_lines="""- fit: the leg silhouette, e.g. "Skinny", "Straight", "Wide", "Bootcut", "Tapered", "Flared".
- length: how much leg it covers, e.g. "Full", "Capri", "Shorts", "Cropped".
- waistRise: where the waistband sits, e.g. "High", "Mid", "Low".
- closure: list every visible fastening mechanism, e.g. "Drawstring", "Elastic", "Zipper", "Buttons", "Hook" — an item can genuinely have more than one (drawstring AND elastic), so include all that apply; use an empty list if none is visible."""
    ),

    # ---- Dresswear -> DresswearMetadata(length, neckline, sleeveType, fit) ----
    Category.DRESSWEAR: _build(
        schema_lines="""{
  "length": "<string>",
  "neckline": "<string>",
  "sleeveType": "<string>",
  "fit": "<string>"
}""",
        guidance_lines="""- length: hemline relative to the body, e.g. "Midi", "Mini", "Maxi".
- neckline: the neckline shape, e.g. "Halter", "Round", "Off-shoulder", "Sweetheart", "V-neck", "One Shoulder".
- sleeveType: sleeve cut, e.g. "Sleeveless", "Full", "Half", "Cap Sleeve", "Three Quarter".
- fit: overall silhouette, e.g. "A-line", "Bodycon", "Flowy", "Wrap", "Fit and Flare", "Shift"."""
    ),

    # ---- Drapewear -> DrapewearMetadata(drapeType, blouseIncluded, workType, fabricWeight) ----
    Category.DRAPEWEAR: _build(
        schema_lines="""{
  "drapeType": "<string>",
  "blouseIncluded": <bool>,
  "workType": "<string>",
  "fabricWeight": "<string>"
}""",
        guidance_lines="""- drapeType: the specific draped garment style, e.g. "Saree", "Lehenga", "Half-saree".
- blouseIncluded: true only if a separate blouse/top is visibly part of the set in the image.
- workType: the embellishment or embroidery style, e.g. "Zari", "Plain", "Embroidered", "Printed", "Sequinned".
- fabricWeight: how heavy the drape looks, e.g. "Heavy" (structured, richly worked) or "Light" (flowy, sheer)."""
    ),

    # ---- Outerwear -> OuterwearMetadata(closureType, warmthLevel) ----
    Category.OUTERWEAR: _build(
        schema_lines="""{
  "closureType": "<string>",
  "warmthLevel": "<string>"
}""",
        guidance_lines="""- closureType: how the front fastens, e.g. "Zip", "Button", "Open" (no fastening, worn loose).
- warmthLevel: how insulating it looks based on material and thickness, e.g. "Light", "Medium", "Heavy"."""
    ),

    Category.INNERWEAR: "",

    # ---- Footwear -> FootwearMetadata(heelHeight, closureType, soleType) ----
    Category.FOOTWEAR: _build(
        schema_lines="""{
  "heelHeight": "<string>",
  "closureType": "<string>",
  "soleType": "<string>"
}""",
        guidance_lines="""- heelHeight: e.g. "Flat", "Low", "Medium", "High".
- closureType: how the shoe stays on the foot, e.g. "Laces", "Velcro", "Slip-on", "Buckle".
- soleType: the visible sole material, e.g. "Rubber", "Leather", "Foam"."""
    ),

    # ---- Hosiery -> HosieryMetadata(length, thickness) ----
    Category.HOSIERY: _build(
        schema_lines="""{
  "length": "<string>",
  "thickness": "<string>"
}""",
        guidance_lines="""- length: how far up the leg it goes, e.g. "Knee-high", "Ankle", "Thigh-high".
- thickness: fabric density, e.g. "Sheer", "Regular", "Thermal"."""
    ),

    Category.HEADWEAR: "",

    # ---- Eyewear -> EyewearMetadata(frameShape, lensType) ----
    Category.EYEWEAR: _build(
        schema_lines="""{
  "frameShape": "<string>",
  "lensType": "<string>"
}""",
        guidance_lines="""- frameShape: the frame's silhouette, e.g. "Cat-eye", "Round", "Square", "Aviator", "Rectangle", "Oversized".
- lensType: what the lenses appear to be for, e.g. "Sunglasses" (visibly tinted/dark), "Prescription" (clear), "Blue-light"."""
    ),

    # ---- Earwear -> EarwearMetadata(closure) ----
    Category.EARWEAR: _build(
        schema_lines="""{
  "closure": "<string>"
}""",
        guidance_lines="""- closure: how the earring attaches, e.g. "Push-back", "Screw-back", "Hook", "Clip-on"."""
    ),

    # ---- Neckwear -> NeckwearMetadata(length) ----
    Category.NECKWEAR: _build(
        schema_lines="""{
  "length": "<string>"
}""",
        guidance_lines="""- length: how the piece sits relative to the neck/chest, e.g. "Choker" (tight to the neck), "Medium", "Long" (falls below the collarbone)."""
    ),

    # ---- Wristwear -> WristwearMetadata(strapMaterial) ----
    Category.WRISTWEAR: _build(
        schema_lines="""{
  "strapMaterial": "<string|null>"
}""",
        guidance_lines="""- strapMaterial: what the band/strap is made of, e.g. "Leather", "Metal", "Fabric". Use null if the item has no strap at all (e.g. a solid bangle)."""
    ),

    # ---- Anklewear -> AnklewearMetadata(hasBells) ----
    Category.ANKLEWEAR: _build(
        schema_lines="""{
  "hasBells": <bool>
}""",
        guidance_lines="""- hasBells: true only if small bells/ghungroo are visibly attached to the anklet."""
    ),

    # ---- Bags -> BagsMetadata(strapType, compartments) ----
    Category.BAGS: _build(
        schema_lines="""{
  "strapType": "<string>",
  "compartments": "<string>"
}""",
        guidance_lines="""- strapType: how it's meant to be carried, e.g. "Crossbody" (long strap worn diagonally), "Handheld" (no long strap), "Shoulder" (single shoulder-length strap).
- compartments: whether multiple distinct visible sections/pockets exist, e.g. "Multiple", "Single"."""
    ),

    # ---- Waistwear -> WaistwearMetadata(buckleType, adjustable) ----
    Category.WAISTWEAR: _build(
        schema_lines="""{
  "buckleType": "<string>",
  "adjustable": <bool>
}""",
        guidance_lines="""- buckleType: the fastening mechanism, e.g. "Pin", "Magnetic", "Hook".
- adjustable: true if the item visibly has multiple holes/notches or another way to change its size."""
    ),
}


def get_metadata_prompt(category: Category) -> str:
    """Return the filled-in Stage-2 prompt for a category, or None if that
    category has no metadata fields (Innerwear, Headwear)."""
    template = CATEGORY_METADATA_PROMPTS.get(category)
    if template is None or template == "":
        return ""
    return template.replace("{category}", category)


if __name__ == "__main__":
    print(BASE_ANALYSIS_PROMPT[:200], "...\n")
    print(get_metadata_prompt(Category.FOOTWEAR))