// =========================================
// WARDROBE APP - TYPESCRIPT SCHEMA (v3)
// =========================================
//
// Design notes:
// 1. `category` stays a literal union — this is structural, not a
//    "guessable value" enum. It's what powers the discriminated union
//    below (TS uses it to know which `metadata` shape applies). Every
//    other classification field uses `string` since an AI model or a
//    user could produce a value you didn't anticipate.
// 2. Any field that repeated across multiple categories (like "type":
//    Jacket/Sneakers/Necklace/Payal/Tote/Belt...) has been pulled out
//    to the common `subType` field on the base item, instead of being
//    redefined inside every metadata block.
// 3. colors / image / confidence are now structured objects, matching
//    what a vision model would realistically output.
// 4. Every field has an inline example next to it.

// ---------- Shared Types ----------

export type Category =
  | "Topwear"
  | "Bottomwear"
  | "Dresswear"
  | "Drapewear"
  | "Outerwear"
  | "Innerwear"
  | "Footwear"
  | "Hosiery"
  | "Headwear"
  | "Eyewear"
  | "Earwear"
  | "Neckwear"
  | "Wristwear"
  | "Anklewear"
  | "Bags"
  | "Waistwear";

export interface ColorInfo {
  name: string; // "Navy Blue"
  rgb: [number, number, number]; // [28, 45, 92]
  percentage: number; // 82   (0-100, how much of the item this color covers)
}

export interface Colors {
  dominant: ColorInfo[]; // [{ name: "Navy Blue", rgb: [28,45,92], percentage: 82 }, ...]
  primary_color_family: string; // "Blue"
  secondary_color_family?: string; // "Neutral"
}

export interface ImageInfo {
  original_url: string; // "https://cdn.app.com/items/123/original.jpg"
  thumbnail_url: string; // "https://cdn.app.com/items/123/thumb.jpg"
  segmented_url?: string; // "https://cdn.app.com/items/123/segmented.png" (background removed)
}

// Confidence scores from whatever model classified the item.
// Kept as a loose record (+ an explicit `overall`) since different
// categories will have different sets of scored attributes
// (e.g. "fabric" makes sense for clothes, not for a watch).
export interface ConfidenceScores {
  overall?: number; // 0.95
  category?: number; // 0.99
  color?: number; // 0.98
  fabric?: number; // 0.83
  pattern?: number; // 0.96
  fit?: number; // 0.91
  [key: string]: number | undefined; // e.g. heelHeight: 0.88, frameShape: 0.9 — for category-specific scores
}

// ---------- Common Attributes (apply to every item) ----------

interface BaseWardrobeItem {
  id: string; // "itm_9f3a1c"
  name: string; // "Blue Denim Jacket"
  category: Category; // "Outerwear"

  // The specific type within the category — e.g. Outerwear -> "Jacket",
  // Bags -> "Tote", Neckwear -> "Necklace", Anklewear -> "Payal".
  subType?: string; // "Jacket"

  colors: Colors; // see Colors interface above
  material: string; // "Denim" (clothes) / "Sterling Silver" (jewelry) / "Leather" (bags)

  // NOTE: kept as string[] since JSON has no native Set type.
  // Dedupe on write (e.g. `[...new Set(season)]`) if you want to
  // guarantee uniqueness before saving.
  season: string[]; // ["Winter", "Monsoon"]
  occasion: string[]; // ["Casual", "Party"]

  image: ImageInfo; // see ImageInfo interface above
  confidence?: ConfidenceScores; // see ConfidenceScores interface above
}

// ---------- Category-Specific Metadata ----------
// Only fields that are genuinely unique to that category live here now.

export interface TopwearMeta {
  category: "Topwear";
  metadata: {
    sleeveType: string; // "Full" | "Half" | "Sleeveless" | "3-quarter"
    neckline: string; // "Round" | "V-neck" | "Collar" | "Boat neck"
    fit: string; // "Slim" | "Loose" | "Regular"
    pattern: string; // "Striped" | "Solid" | "Printed" | "Checked"
    length: string; // "Regular" | "Crop" | "Longline"
  };
}

export interface BottomwearMeta {
  category: "Bottomwear";
  metadata: {
    fit: string; // "Skinny" | "Straight" | "Wide" | "Bootcut"
    length: string; // "Full" | "Capri" | "Shorts" | "Cropped"
    waistRise: string; // "High" | "Mid" | "Low"
    closure: string; // "Zipper" | "Button" | "Drawstring" | "Elastic" this should be a array of strings because zipper comes with not only button and also drawstring come with elastic but in some pants it's only elastic or drawstring.
  };
}

export interface DresswearMeta {
  category: "Dresswear";
  metadata: {
    length: string; // "Midi" | "Mini" | "Maxi"
    neckline: string; // "Halter" | "Round" | "Off-shoulder"
    sleeveType: string; // "Sleeveless" | "Full" | "Half"
    fit: string; // "A-line" | "Bodycon" | "Flowy"
  };
}

export interface DrapewearMeta {
  category: "Drapewear";
  metadata: {
    drapeType: string; // "Saree" | "Lehenga" | "Half-saree"
    blouseIncluded: boolean; // true | false
    workType: string; // "Zari" | "Plain" | "Embroidered" | "Printed"
    fabricWeight: string; // "Heavy" | "Light"
  };
}

export interface OuterwearMeta {
  category: "Outerwear";
  metadata: {
    closureType: string; // "Zip" | "Button" | "Open"
    warmthLevel: string; // "Medium" | "Light" | "Heavy"
  };
}

export interface InnerwearMeta {
  category: "Innerwear";
  metadata?: Record<string, never>; // no extra fields
}

export interface FootwearMeta {
  category: "Footwear";
  metadata: {
    heelHeight: string; // "Flat" | "Low" | "Medium" | "High"
    closureType: string; // "Laces" | "Velcro" | "Slip-on" | "Buckle"
    soleType: string; // "Rubber" | "Leather" | "Foam"
  };
}

export interface HosieryMeta {
  category: "Hosiery";
  metadata: {
    length: string; // "Knee-high" | "Ankle" | "Thigh-high"
    thickness: string; // "Sheer" | "Regular" | "Thermal"
  };
}

export interface HeadwearMeta {
  category: "Headwear";
  metadata?: Record<string, never>; // no extra fields yet beyond subType/common
}

export interface EyewearMeta {
  category: "Eyewear";
  metadata: {
    frameShape: string; // "Cat-eye" | "Round" | "Square" | "Aviator"
    lensType: string; // "Sunglasses" | "Prescription" | "Blue-light"
  };
}

export interface EarwearMeta {
  category: "Earwear";
  metadata: {
    closure: string; // "Push-back" | "Screw-back" | "Hook"
  };
}

export interface NeckwearMeta {
  category: "Neckwear";
  metadata: {
    length: string; // "Choker" | "Medium" | "Long"
  };
}

export interface WristwearMeta {
  category: "Wristwear";
  metadata: {
    strapMaterial?: string; // "Leather" | "Metal" | "Fabric"  (watches only)
  };
}

export interface AnklewearMeta {
  category: "Anklewear";
  metadata: {
    hasBells: boolean; // true | false
  };
}

export interface BagsMeta {
  category: "Bags";
  metadata: {
    strapType: string; // "Crossbody" | "Handheld" | "Shoulder"
    compartments: string; // "Multiple" | "Single"
  };
}

export interface WaistwearMeta {
  category: "Waistwear";
  metadata: {
    buckleType: string; // "Pin" | "Magnetic" | "Hook"
    adjustable: boolean; // true | false
  };
}

// ---------- Discriminated Union of All Category Metadata ----------

export type CategoryMeta =
  | TopwearMeta
  | BottomwearMeta
  | DresswearMeta
  | DrapewearMeta
  | OuterwearMeta
  | InnerwearMeta
  | FootwearMeta
  | HosieryMeta
  | HeadwearMeta
  | EyewearMeta
  | EarwearMeta
  | NeckwearMeta
  | WristwearMeta
  | AnklewearMeta
  | BagsMeta
  | WaistwearMeta;

// ---------- Final WardrobeItem Type ----------

export type WardrobeItem = Omit<BaseWardrobeItem, "category"> & CategoryMeta;

// =========================================
// USAGE EXAMPLES
// =========================================

const jacket: WardrobeItem = {
  id: "1",
  name: "Blue Denim Jacket",
  category: "Outerwear",
  subType: "Jacket",
  colors: {
    dominant: [
      { name: "Denim Blue", rgb: [55, 78, 110], percentage: 95 },
      { name: "White", rgb: [245, 245, 245], percentage: 5 },
    ],
    primary_color_family: "Blue",
  },
  material: "Denim",

  season: ["Winter", "Monsoon"],
  occasion: ["Casual"],
  image: {
    original_url: "https://example.com/jacket-original.jpg",
    thumbnail_url: "https://example.com/jacket-thumb.jpg",
    segmented_url: "https://example.com/jacket-segmented.png",
  },
  confidence: {
    overall: 0.95,
    category: 0.99,
    color: 0.97,
    fabric: 0.9,
  },

  metadata: {
    closureType: "Zip",
    warmthLevel: "Medium",
  },
};

const saree: WardrobeItem = {
  id: "2",
  name: "Red Silk Saree",
  category: "Drapewear",
  subType: "Saree",
  colors: {
    dominant: [
      { name: "Navy Blue", rgb: [28, 45, 92], percentage: 82 },
      { name: "White", rgb: [245, 245, 245], percentage: 18 },
    ],
    primary_color_family: "Blue",
    secondary_color_family: "Neutral",
  },
  material: "Silk",
  season: ["Summer", "Winter", "Monsoon"],
  occasion: ["Ethnic", "Party"],
  image: {
    original_url: "https://example.com/saree-original.jpg",
    thumbnail_url: "https://example.com/saree-thumb.jpg",
  },
  confidence: {
    overall: 0.93,
    category: 0.99,
    color: 0.98,
    fabric: 0.83,
    pattern: 0.96,
  },

  metadata: {
    drapeType: "Saree",
    blouseIncluded: true,
    workType: "Zari",
    fabricWeight: "Heavy",
  },
};

const sneakers: WardrobeItem = {
  id: "3",
  name: "White Casual Sneakers",
  category: "Footwear",
  subType: "Sneakers",
  colors: {
    dominant: [{ name: "White", rgb: [250, 250, 250], percentage: 100 }],
    primary_color_family: "White",
  },
  material: "Canvas",

  season: ["All-season"],
  occasion: ["Casual"],
  image: {
    original_url: "https://example.com/sneakers-original.jpg",
    thumbnail_url: "https://example.com/sneakers-thumb.jpg",
  },
  confidence: {
    overall: 0.97,
    category: 0.99,
    color: 0.95,
  },
  metadata: {
    heelHeight: "Flat",
    closureType: "Laces",
    soleType: "Rubber",
  },
};

// TS still catches structural mistakes — e.g. putting Drapewear's
// `metadata` shape on a Footwear item — even though the individual
// field values inside metadata are now plain strings.
// Example (uncomment to see the error):
// const invalidItem: WardrobeItem = {
//   ...jacket,
//   category: "Footwear",
//   metadata: {
//     drapeType: "Saree", // ❌ Error: not a valid Footwear metadata field
//   },
// };
