# Wardly — AI-Powered Wardrobe & Outfit Stylist

Wardly is an intelligent wardrobe management and outfit suggestion system. Upload photos of your clothing items, and the system automatically analyses them — extracting colour palettes, fabric types, patterns, fit, and other fashion metadata using Gemini AI vision. It then generates a 512-dimension FashionCLIP embedding for each item and stores everything in MongoDB Atlas. When you describe an occasion ("a rooftop party", "Sunday brunch"), Wardly performs a semantic vector search across your closet and uses Gemini to compose 1–2 complete outfit combinations from the best-matching pieces.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Backend API](#backend-api)
  - [POST /api/items — Upload a Wardrobe Item](#post-apiitems--upload-a-wardrobe-item)
  - [POST /api/outfit-suggestions — Get Outfit Ideas](#post-apioutfit-suggestions--get-outfit-ideas)
- [AI Pipelines](#ai-pipelines)
  - [Image Ingestion Pipeline](#image-ingestion-pipeline)
  - [Outfit Suggestion Pipeline](#outfit-suggestion-pipeline)
- [Database Schema](#database-schema)
- [Environment Variables](#environment-variables)
- [Prerequisites](#prerequisites)
- [Setup & Running](#setup--running)
- [Web UI](#web-ui)

---

## Architecture Overview

```
                         ┌──────────────────────────┐
                         │      Upload (HTML)       │
                         │   Outfits Query (HTML)   │
                         └──────────┬───────────────┘
                                    │ HTTP
                         ┌──────────▼───────────────┐
                         │   FastAPI Backend         │
                         │   (uvicorn)               │
                         └──┬──────────────┬────────┘
                            │              │
                    ┌───────▼──────┐  ┌────▼──────────┐
                    │   Gemini AI  │  │    MongoDB     │
                    │  (Vision +   │  │   Atlas        │
                    │   Text Gen)  │  │  (Vector       │
                    └──────────────┘  │   Search +     │
                                      │   Documents)   │
                    ┌──────────────┐  └────┬──────────┘
                    │  FashionCLIP │       │
                    │  (Embedding) │  ┌────▼──────────┐
                    └──────────────┘  │   Cloudinary   │
                    ┌──────────────┐  │ (Image Host)  │
                    │   Segformer  │  └───────────────┘
                    │ (Garment     │
                    │  Segmentation)│
                    └──────────────┘
                    ┌──────────────┐
                    │   rembg      │
                    │ (Background  │
                    │  Removal)    │
                    └──────────────┘
```

1. **User uploads a clothing photo** via the upload form or API.
2. **Gemini Vision** classifies the image type (`product_flat`, `product_mannequin`, `on_model`, `unusable`) and detects the garment category.
3. **Segformer** (for on-model photos) or **rembg** (for flat/mannequin shots) extracts the garment as a transparent PNG cutout.
4. **Gemini Vision** analyses the cutout for detailed fashion attributes: category, subcategory, fit, sleeve length, collar style, pattern, fabric, texture, closure, dominant colours with hex/RGB/HSV values, and confidence scores.
5. **FashionCLIP** generates a 512-dimension unit-normalised embedding of the cutout image.
6. **Cloudinary** hosts the original image, the segmented cutout, and the mask. Thumbnails are served on-the-fly via Cloudinary URL transformations.
7. A composite document is saved to **MongoDB Atlas** under the `wardrobe_items` collection.
8. When the user describes an occasion, the query text is embedded with **FashionCLIP's text encoder** (same 512-dim joint space), and a **MongoDB Atlas `$vectorSearch`** finds the closest-matching items. **Gemini** then composes 1–2 outfits from the shortlist.

---

## Features

- **Automatic Image Classification** – Gemini determines whether the photo is a flat lay, mannequin shot, on-model, or unusable.
- **Garment Segmentation** – Segformer (fine-grained human parsing) for on-model photos; rembg (generic foreground removal) for flat/mannequin shots.
- **Rich Fashion Metadata Extraction** – Colour families, dominant colours (hex/RGB/HSV), pattern, fabric, texture, fit, sleeve length, neck style, closure type, and more — all extracted by Gemini.
- **FashionCLIP Embeddings** – 512-dim joint text-image embedding space enabling semantic outfit search.
- **MongoDB Atlas Vector Search** – Cosine-similarity vector search over the user's wardrobe, filtered by owner.
- **AI Outfit Composition** – Gemini receives the vector-search shortlist and returns 1–2 outfit combinations with styling notes and per-item reasoning.
- **Cloudinary Integration** – Automatic image hosting with on-the-fly thumbnail generation.
- **Two Static Frontends** – A minimal upload form (`/`) and a styled outfit-query interface (`/outfits.html`).

---

## Tech Stack

| Layer                  | Technology                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **Runtime**            | Python ≥3.12                                                                       |
| **Framework**          | FastAPI + uvicorn                                                                  |
| **Database**           | MongoDB Atlas (with `$vectorSearch` Atlas Search index)                            |
| **Vision AI**          | Google Gemini 2.5 Flash (classification) + Gemini 3.5 Flash (attribute extraction) |
| **Text Generation**    | Google Gemini 3.5 Flash (outfit composition)                                       |
| **Image Embedding**    | Marqo-FashionCLIP via `open_clip` (hf-hub:Marqo/marqo-fashionCLIP)                 |
| **Segmentation**       | SegFormer B2 (`mattmdjaga/segformer_b2_clothes`) for human parsing                 |
| **Background Removal** | rembg (CPU)                                                                        |
| **Image Hosting**      | Cloudinary                                                                         |
| **ML / Data**          | PyTorch, NumPy, Pillow, scikit‑image                                               |
| **Package Manager**    | `uv`                                                                               |

---

## Project Structure

```
wardly/
├── README.md                     # This file
├── backend/
│   ├── .env                      # Environment variables (see below)
│   ├── .gitignore
│   ├── .python-version
│   ├── pyproject.toml            # Project config + dependencies
│   ├── uv.lock                   # Locked dependency versions
│   ├── main.py                   # FastAPI app, routes, CORS
│   ├── ai_agent.py               # Outfit suggestion orchestrator
│   ├── wardrobe_image_processor.py  # Image pipeline + DB helpers
│   ├── wardrobe_image_processor.py  # (duplicate reference in listing)
│   ├── data.json                 # Sample wardrobe document
│   ├── item.json                 # Sample item document
│   ├── item copy.json
│   ├── item-pipeline.txt
│   ├── yolo11n-seg.pt            # YOLO segmentation model (optional)
│   ├── enums/
│   │   └── clothing_item.py      # Fashion enums (Category, Fabric, Fit, etc.)
│   ├── static/
│   │   ├── index.html            # Minimal upload page
│   │   └── outfits.html          # Outfit query UI
│   ├── output/                   # Generated garment cutouts (PNG)
│   ├── cloths/                   # Training / test images
│   └── tmp/                      # Temporary uploads (gitignored)
```

---

## Backend API

### POST /api/items — Upload a Wardrobe Item

Upload a clothing photo and get it processed, analysed, embedded, and stored.

**Request** (multipart/form-data):

| Field      | Type          | Required | Default      | Description      |
| ---------- | ------------- | -------- | ------------ | ---------------- |
| `file`     | UploadFile    | Yes      | —            | Clothing photo   |
| `owner_id` | string (Form) | No       | `"user_123"` | Owner identifier |

**Response** (201 Created):

```json
{
  "_id": "663f1a2b...",
  "owner_id": "user_123",
  "vision_metadata": {
    "colors": {
      "dominant": [
        { "name": "Rust Brown", "rgb": [166, 88, 30], "hsv": [26, 82, 65],
          "hex": "#A6581E", "percentage": 82 },
        { "name": "White", "rgb": [255, 255, 255], "hsv": [0, 0, 100],
          "hex": "#FFFFFF", "percentage": 18 }
      ],
      "primary_color_family": "Brown",
      "secondary_color_family": "White"
    },
    "detected_attributes": {
      "category": "Shirt",
      "subcategory": "Button-down",
      "fit": "Regular Fit",
      "sleeve_length": "Full Sleeve",
      "neck_style": "Button Down",
      "pattern": "Solid",
      "fabric": "Cotton",
      "texture": "Smooth",
      "closure": "Buttons",
      "logo_present": false,
      "graphic_present": false,
      "hood": false,
      "pockets": 2
    },
    "confidence": {
      "category": 0.98, "color": 0.95, "fabric": 0.8,
      "pattern": 0.99, "fit": 0.9
    },
    "image": {
      "original_url": "https://res.cloudinary.com/.../original.jpg",
      "thumbnail_url": "https://res.cloudinary.com/.../w_300/...",
      "segmented_url": "https://res.cloudinary.com/.../cutout.png"
    },
    "segmentation": {
      "mask_url": "https://res.cloudinary.com/.../mask.png",
      "bounding_box": [120, 45, 410, 560]
    },
    "embedding": [0.023, -0.041, ...]  // 512-dim float vector
  }
}
```

**Error codes:**

| Status | Condition                                          |
| ------ | -------------------------------------------------- |
| 422    | Image is unusable (blurry, no garment visible)     |
| 400    | Unsupported garment category for on-model parsing  |
| 503    | Gemini analysis service is temporarily unavailable |

---

### POST /api/outfit-suggestions — Get Outfit Ideas

Describe an occasion; receive 1–2 composed outfits from your wardrobe.

**Request** (application/json):

```json
{
  "query": "I want to go to a rooftop party",
  "owner_id": "user_123"
}
```

**Response** (200 OK):

```json
{
  "combinations": [
    {
      "title": "Rooftop Party Chic",
      "items": [
        {
          "item_id": "663f1a2b...",
          "category": "Shirt",
          "reason": "The rust-brown button-down adds a warm, sophisticated base that's perfect for evening events.",
          "image_url": "https://res.cloudinary.com/.../w_300/..."
        },
        {
          "item_id": "663f1a2c...",
          "category": "Jeans",
          "reason": "Dark-wash jeans keep the look grounded and smart-casual.",
          "image_url": "https://res.cloudinary.com/.../w_300/..."
        }
      ],
      "styling_notes": "Roll the sleeves for a relaxed vibe. Pair with minimalist sneakers or loafers."
    }
  ]
}
```

**Error codes:**

| Status | Condition                                               |
| ------ | ------------------------------------------------------- |
| 404    | No wardrobe items found for this user                   |
| 502    | Gemini referenced an item that was not in the shortlist |
| 503    | Styling service is temporarily unavailable              |

---

## AI Pipelines

### Image Ingestion Pipeline

```
Upload Photo
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Gemini 2.5 Flash — Image Classification              │
│    - Determines: product_flat / product_mannequin /     │
│      on_model / unusable                                │
│    - Detects: garment category (shirt, pants, dress…)   │
│    - If unusable → rejects with UnusableImageError     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
              ┌───────────┴───────────┐
              │                       │
    on_model  │                       │  product_flat /
              │                       │  product_mannequin
              ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│ 2a. Segformer B2     │  │ 2b. rembg            │
│     (human parsing)  │  │     (foreground       │
│     - Isolates the   │  │      removal)         │
│       specific       │  │     - Whole foreground│
│       garment by     │  │       → cutout        │
│       label          │  │                       │
│     - Raises         │  │                       │
│       Unsupported-   │  │                       │
│       CategoryError  │  │                       │
│       if mapping     │  │                       │
│       missing        │  │                       │
└──────────┬───────────┘  └──────────┬────────────┘
           │                         │
           └──────────┬──────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Save transparent PNG cutout + mask locally           │
│    Outputs: cutout_path, mask_path, bounding_box        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Upload to Cloudinary                                 │
│    - Original image     → original_url                  │
│    - Cutout PNG         → segmented_url                 │
│    - Mask               → mask_url                      │
│    - Thumbnail (on-the-fly url transform)               │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌────────────────────────┐  ┌────────────────────────────┐
│ 5a. Gemini 3.5 Flash   │  │ 5b. FashionCLIP            │
│     — Attribute        │  │     — Vision Embedding     │
│     Extraction         │  │                            │
│     - Colour analysis  │  │     512-dim unit-normalised│
│     - Category, fit,   │  │     embedding for vector    │
│       sleeve, neck…    │  │     search                 │
│     - Pattern, fabric, │  │                            │
│       texture, closure │  │                            │
│     - Confidence scores│  │                            │
└──────────┬─────────────┘  └─────────────┬──────────────┘
           └──────────────┬────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Build & save document                                │
│    build_wardrobe_document(...) → MongoDB insert_one    │
└─────────────────────────────────────────────────────────┘
```

### Outfit Suggestion Pipeline

```
User Query (e.g. "rooftop party")
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Embed query with FashionCLIP Text Encoder            │
│    512-dim unit-normalised vector, same space as images │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. MongoDB Atlas $vectorSearch                          │
│    - Index: wardrobe_vector_index                       │
│    - Path: vision_metadata.embedding                    │
│    - Metric: cosine                                     │
│    - Filter: { owner_id: "user_123" }                   │
│    - Returns: top-15 items with vectorSearchScore       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Gemini 3.5 Flash — Outfit Composition                │
│    - Prompt: build 1-2 outfits using ONLY these items   │
│    - Input: shortlist JSON with categories, colors,     │
│      attributes, fashion_metadata                       │
│    - Structured output via Pydantic schema:             │
│      OutfitSuggestions.combinations[]                   │
│        → title, items[item_id, category, reason,       │
│          image_url], styling_notes                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Post-processing                                      │
│    - Validates all referenced item_ids exist in         │
│      shortlist                                          │
│    - Injects image_url (thumbnail) from stored doc      │
│    - Falls back to direct Mongo lookup if thumbnail     │
│      missing from projection                            │
│    - Raises ValueError if hallucinated item detected    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
                    Returned to client
```

---

## Database Schema

### MongoDB Collection: `wardly.wardrobe_items`

```json
{
  "_id": "ObjectId(...)",
  "owner_id": "user_123",
  "vision_metadata": {
    "colors": {
      "dominant": [
        {
          "name": "string",
          "rgb": [int, int, int],
          "hsv": [int, int, int],
          "hex": "#RRGGBB",
          "percentage": int
        }
      ],
      "primary_color_family": "string",
      "secondary_color_family": "string"
    },
    "detected_attributes": {
      "category": "string (enum: Category)",
      "subcategory": "string",
      "fit": "string (enum: Fit)",
      "sleeve_length": "string (enum: SleeveLength)",
      "neck_style": "string (enum: NeckStyle)",
      "pattern": "string (enum: Pattern)",
      "fabric": "string (enum: Fabric)",
      "texture": "string (enum: Texture)",
      "closure": "string (enum: Closure)",
      "neckline": "string | null",
      "logo_present": false,
      "graphic_present": false,
      "hood": false,
      "pockets": 0
    },
    "confidence": {
      "category": 0.0–1.0,
      "color": 0.0–1.0,
      "fabric": 0.0–1.0,
      "pattern": 0.0–1.0,
      "fit": 0.0–1.0
    },
    "image": {
      "original_url": "https://res.cloudinary.com/...",
      "thumbnail_url": "https://res.cloudinary.com/.../w_300/...",
      "segmented_url": "https://res.cloudinary.com/..."
    },
    "segmentation": {
      "mask_url": "https://res.cloudinary.com/...",
      "bounding_box": [x0, y0, x1, y1]
    },
    "embedding": [0.001, -0.042, ...]  // 512-dim float array
  }
}
```

### Required Atlas Search Index

A MongoDB Atlas Search index named **`wardrobe_vector_index`** must exist on the `wardrobe_items` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "vision_metadata.embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "owner_id"
    }
  ]
}
```

---

## Environment Variables

Create a `.env` file in the `backend/` directory:

```
GEMINI_API_KEY=your_gemini_api_key
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGODB_DB=wardly
```

---

## Prerequisites

- **Python** ≥3.12
- **uv** (Python package manager) — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **MongoDB Atlas** cluster with Search index configured (see [Database Schema](#database-schema))
- **Google Gemini API** key (with access to `gemini-2.5-flash` and `gemini-3.5-flash` models)
- **Cloudinary** account (free tier sufficient) for image hosting
- Sufficient RAM for FashionCLIP, Segformer, and PyTorch models (~4–6 GB)

---

## Setup & Running

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/wardly.git
cd wardly/backend

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your keys

# 3. Install dependencies (creates virtual environment automatically)
uv sync

# 4. Start the development server
uv run --env-file .env uvicorn main:app --reload

# The API will be available at http://localhost:8000
# Upload page: http://localhost:8000/
# Outfits query: http://localhost:8000/outfits.html
# API docs: http://localhost:8000/docs
```

> **Note on model loading:** FashionCLIP, Segformer, and their tokenizers/processors are loaded lazily on first use, so the first upload or query may take 10–30 seconds while PyTorch downloads and caches the model weights. Subsequent requests will be fast.

---

## Web UI

Two static frontends are served from the FastAPI backend:

### `/` — Upload Page (`static/index.html`)

Minimal form to upload a clothing photo. Displays the saved wardrobe document as JSON.

### `/outfits.html` — Outfit Query Page

A styled interface for describing an occasion and viewing outfit suggestions. Features:

- Natural-language query input
- Configurable owner/closet ID
- Example chips for quick testing ("a party", "weekend brunch", "first date", "work presentation")
- Animated result cards with item thumbnails, per-item reasoning, and styling notes

---

## Development Notes

- **`_mongo_safe()`** recursively converts MongoDB `ObjectId` instances to strings for JSON-safe serialisation.
- **Lazy model loading** is used for FashionCLIP, Segformer, and the FashionCLIP tokenizer — singletons are cached at module level for the process lifetime.
- **MongoClient** is also lazily initialised as a process-wide singleton (thread-safe connection pooling).
- The `owner_id` default of `"user_123"` is a placeholder; replace with real authentication logic before production.
- CORS is currently configured with `allow_origins=["*"]` — tighten to your frontend domain before shipping.
- Static files are mounted AFTER dynamic API routes so that `/api/` paths take precedence over the catch-all static mount.
