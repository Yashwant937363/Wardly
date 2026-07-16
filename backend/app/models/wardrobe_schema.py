"""
Wardrobe item schema — Pydantic parallel of the TypeScript version.

Design notes (mirrors the TS file's reasoning):
1. `category` is a Literal per subclass, not a free string — this is what
   drives the discriminated union below (Pydantic uses Field(discriminator=...)
   the same way TS uses the literal to narrow which `metadata` shape applies).
   Every other classification field is `str`, since an AI model or a user
   could produce a value that wasn't anticipated.
2. Fields that repeat across categories (e.g. "type": Jacket/Sneakers/
   Necklace/Payal/Tote/Belt) live on the shared `subType` field instead of
   being redefined inside every metadata block.
3. `embedding` is present here (was missing in the TS draft) — the 512-dim
   FashionCLIP vector every vector-search query depends on.
4. `fashion_metadata` is intentionally NOT reintroduced. weather/layering/
   dress_code/compatible_* were dropped earlier because neither the model
   nor the user can reliably determine them — only `season` and `occasion`
   remain, both as top-level fields.
5. No standalone `id` field — MongoDB's own `_id` is the document's
   identity. This model represents what gets BUILT before insertion;
   `_id` gets attached by pymongo at insert time, same as the current
   save_wardrobe_item().
6. Only RGB is stored for color, not RGB+HSV+hex — one format is enough,
   storing the same color three ways just bloats every document.
7. user_metadata (favorite/rating/laundry/purchase/usage) is deliberately
   OMITTED for this phase — scope right now is outfit suggestion only,
   not full wardrobe management. Add it back when that phase starts.
8. Any field that can genuinely hold more than one value at once (e.g. a
   pant with both a drawstring AND an elastic waist) is a list[str], not
   a single str — matches the TS array fields exactly.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------- Shared Types ----------


class Category(str, Enum):
    TOPWEAR = "Topwear"
    BOTTOMWEAR = "Bottomwear"
    DRESSWEAR = "Dresswear"
    DRAPEWEAR = "Drapewear"
    OUTERWEAR = "Outerwear"
    INNERWEAR = "Innerwear"
    FOOTWEAR = "Footwear"
    HOSIERY = "Hosiery"
    HEADWEAR = "Headwear"
    EYEWEAR = "Eyewear"
    EARWEAR = "Earwear"
    NECKWEAR = "Neckwear"
    WRISTWEAR = "Wristwear"
    ANKLEWEAR = "Anklewear"
    BAGS = "Bags"
    WAISTWEAR = "Waistwear"


class ColorInfo(BaseModel):
    name: str  # "Navy Blue" | "Maroon" | "Off-white"
    rgb: tuple[int, int, int]  # (28, 45, 92)
    percentage: int  # 0-100, how much of the item this color covers


class Colors(BaseModel):
    dominant: list[ColorInfo]
    primary_color_family: str  # "Blue" | "Red" | "Neutral" | "Black"
    secondary_color_family: str | None = None


class ImageInfo(BaseModel):
    original_url: str
    thumbnail_url: str
    segmented_url: str | None = None


class ConfidenceScores(BaseModel):
    """
    Loose by design: named fields for the common ones, but extra="allow"
    lets category-specific confidence scores (heelHeight, frameShape,
    buckleType, ...) pass through without needing a new field defined
    for every possible attribute — mirrors the TS index signature
    `[key: string]: number | undefined`.
    """

    model_config = ConfigDict(extra="allow")

    overall: float | None = None
    category: float | None = None
    color: float | None = None
    fabric: float | None = None
    pattern: float | None = None
    fit: float | None = None


# ---------- Shared Base (every item has these) ----------


class WardrobeItemBase(BaseModel):
    name: str  # "Blue Denim Jacket"
    subType: str | None = None  # "Jacket" | "Tote" | "Necklace" | "Payal"

    colors: Colors
    material: str  # "Denim" | "Sterling Silver" | "Cotton"

    season: list[str] = []  # ["Winter", "Monsoon"]
    occasion: list[str] = []  # ["Casual", "Party"]

    image: ImageInfo
    confidence: ConfidenceScores | None = None

    embedding: list[float]  # 512-dim FashionCLIP image embedding


# ---------- Category-Specific Metadata ----------
# Only fields genuinely unique to that category live here.


class TopwearMetadata(BaseModel):
    sleeveType: str  # "Full" | "Half" | "Sleeveless" | "3-quarter"
    neckline: str  # "Round" | "V-neck" | "Collar" | "Boat neck"
    fit: str  # "Slim" | "Loose" | "Regular"
    pattern: str  # "Striped" | "Solid" | "Printed" | "Checked"
    length: str  # "Regular" | "Crop" | "Longline"


class BottomwearMetadata(BaseModel):
    fit: str  # "Skinny" | "Straight" | "Wide" | "Bootcut"
    length: str  # "Full" | "Capri" | "Shorts" | "Cropped"
    waistRise: str  # "High" | "Mid" | "Low"
    closure: list[str] = []  # ["Drawstring", "Elastic"] — can co-occur


class DresswearMetadata(BaseModel):
    length: str  # "Midi" | "Mini" | "Maxi"
    neckline: str  # "Halter" | "Round" | "Off-shoulder"
    sleeveType: str  # "Sleeveless" | "Full" | "Half"
    fit: str  # "A-line" | "Bodycon" | "Flowy"


class DrapewearMetadata(BaseModel):
    drapeType: str  # "Saree" | "Lehenga" | "Half-saree"
    blouseIncluded: bool
    workType: str  # "Zari" | "Plain" | "Embroidered" | "Printed"
    fabricWeight: str  # "Heavy" | "Light"


class OuterwearMetadata(BaseModel):
    closureType: str  # "Zip" | "Button" | "Open"
    warmthLevel: str  # "Medium" | "Light" | "Heavy"


class InnerwearMetadata(BaseModel):
    pass  # no extra fields


class FootwearMetadata(BaseModel):
    heelHeight: str  # "Flat" | "Low" | "Medium" | "High"
    closureType: str  # "Laces" | "Velcro" | "Slip-on" | "Buckle"
    soleType: str  # "Rubber" | "Leather" | "Foam"


class HosieryMetadata(BaseModel):
    length: str  # "Knee-high" | "Ankle" | "Thigh-high"
    thickness: str  # "Sheer" | "Regular" | "Thermal"


class HeadwearMetadata(BaseModel):
    pass  # no extra fields yet beyond subType/common


class EyewearMetadata(BaseModel):
    frameShape: str  # "Cat-eye" | "Round" | "Square" | "Aviator"
    lensType: str  # "Sunglasses" | "Prescription" | "Blue-light"


class EarwearMetadata(BaseModel):
    closure: str  # "Push-back" | "Screw-back" | "Hook"


class NeckwearMetadata(BaseModel):
    length: str  # "Choker" | "Medium" | "Long"


class WristwearMetadata(BaseModel):
    strapMaterial: str | None = None  # "Leather" | "Metal" | "Fabric" (watches only)


class AnklewearMetadata(BaseModel):
    hasBells: bool


class BagsMetadata(BaseModel):
    strapType: str  # "Crossbody" | "Handheld" | "Shoulder"
    compartments: str  # "Multiple" | "Single"


class WaistwearMetadata(BaseModel):
    buckleType: str  # "Pin" | "Magnetic" | "Hook"
    adjustable: bool


# ---------- Discriminated Union of All Category Items ----------
# Each subclass pairs a category with its matching metadata shape.
# Field(discriminator="category") is Pydantic's equivalent of TS
# narrowing a union by a literal field.


class TopwearItem(WardrobeItemBase):
    category: Literal[Category.TOPWEAR] = Category.TOPWEAR
    metadata: TopwearMetadata


class BottomwearItem(WardrobeItemBase):
    category: Literal[Category.BOTTOMWEAR] = Category.BOTTOMWEAR
    metadata: BottomwearMetadata


class DresswearItem(WardrobeItemBase):
    category: Literal[Category.DRESSWEAR] = Category.DRESSWEAR
    metadata: DresswearMetadata


class DrapewearItem(WardrobeItemBase):
    category: Literal[Category.DRAPEWEAR] = Category.DRAPEWEAR
    metadata: DrapewearMetadata


class OuterwearItem(WardrobeItemBase):
    category: Literal[Category.OUTERWEAR] = Category.OUTERWEAR
    metadata: OuterwearMetadata


class InnerwearItem(WardrobeItemBase):
    category: Literal[Category.INNERWEAR] = Category.INNERWEAR
    metadata: InnerwearMetadata = InnerwearMetadata()


class FootwearItem(WardrobeItemBase):
    category: Literal[Category.FOOTWEAR] = Category.FOOTWEAR
    metadata: FootwearMetadata


class HosieryItem(WardrobeItemBase):
    category: Literal[Category.HOSIERY] = Category.HOSIERY
    metadata: HosieryMetadata


class HeadwearItem(WardrobeItemBase):
    category: Literal[Category.HEADWEAR] = Category.HEADWEAR
    metadata: HeadwearMetadata = HeadwearMetadata()


class EyewearItem(WardrobeItemBase):
    category: Literal[Category.EYEWEAR] = Category.EYEWEAR
    metadata: EyewearMetadata


class EarwearItem(WardrobeItemBase):
    category: Literal[Category.EARWEAR] = Category.EARWEAR
    metadata: EarwearMetadata


class NeckwearItem(WardrobeItemBase):
    category: Literal[Category.NECKWEAR] = Category.NECKWEAR
    metadata: NeckwearMetadata


class WristwearItem(WardrobeItemBase):
    category: Literal[Category.WRISTWEAR] = Category.WRISTWEAR
    metadata: WristwearMetadata


class AnklewearItem(WardrobeItemBase):
    category: Literal[Category.ANKLEWEAR] = Category.ANKLEWEAR
    metadata: AnklewearMetadata


class BagsItem(WardrobeItemBase):
    category: Literal[Category.BAGS] = Category.BAGS
    metadata: BagsMetadata


class WaistwearItem(WardrobeItemBase):
    category: Literal[Category.WAISTWEAR] = Category.WAISTWEAR
    metadata: WaistwearMetadata


WardrobeItem = Annotated[
    Union[
        TopwearItem,
        BottomwearItem,
        DresswearItem,
        DrapewearItem,
        OuterwearItem,
        InnerwearItem,
        FootwearItem,
        HosieryItem,    
        HeadwearItem,
        EyewearItem,
        EarwearItem,
        NeckwearItem,
        WristwearItem,
        AnklewearItem,
        BagsItem,
        WaistwearItem,
    ],
    Field(discriminator="category"),
]


class WardrobeItemAdapter(BaseModel):
    """
    Wrap WardrobeItem so it can be validated directly, e.g.:
        item = WardrobeItemAdapter.model_validate(some_dict).root
    Needed because a bare Annotated[Union[...], Field(discriminator=...)]
    isn't itself a BaseModel you can call .model_validate() on directly
    in every Pydantic v2 usage pattern.
    """

    root: WardrobeItem


# =========================================
# USAGE EXAMPLE (mirrors the TS file's examples)
# =========================================

if __name__ == "__main__":
    jacket = OuterwearItem(
        name="Blue Denim Jacket",
        subType="Jacket",
        colors=Colors(
            dominant=[
                ColorInfo(name="Denim Blue", rgb=(55, 78, 110), percentage=95),
                ColorInfo(name="White", rgb=(245, 245, 245), percentage=5),
            ],
            primary_color_family="Blue",
        ),
        material="Denim",
        season=["Winter", "Monsoon"],
        occasion=["Casual"],
        image=ImageInfo(
            original_url="https://example.com/jacket-original.jpg",
            thumbnail_url="https://example.com/jacket-thumb.jpg",
            segmented_url="https://example.com/jacket-segmented.png",
        ),
        confidence=ConfidenceScores(overall=0.95, category=0.99, color=0.97, fabric=0.9),
        embedding=[0.0] * 512,
        metadata=OuterwearMetadata(closureType="Zip", warmthLevel="Medium"),
    )
    print(jacket.model_dump_json(indent=2))

    # Structural mistakes are still caught — putting Drapewear's metadata
    # shape on a Footwear item fails validation, same guarantee as the TS
    # version, just enforced at runtime instead of compile time.
    try:
        FootwearItem(
            name="bad item",
            colors=jacket.colors,
            material="x",
            image=jacket.image,
            embedding=[0.0] * 512,
            metadata=DrapewearMetadata(  # type: ignore[arg-type]
                drapeType="Saree", blouseIncluded=True, workType="Plain", fabricWeight="Light"
            ),
        )
    except Exception as e:
        print(f"\nExpected validation failure: {e}")