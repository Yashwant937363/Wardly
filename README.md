# Wardly — AI-Powered Wardrobe Management & Outfit Suggestion API

Wardly is a **FastAPI** backend that lets users upload photos of their clothing items, automatically analyzes them using computer vision and AI, stores them in a searchable digital wardrobe, and generates outfit suggestions based on natural-language queries.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [Upload a Wardrobe Item](#1-upload-a-wardrobe-item)
  - [List All Items](#2-list-all-items)
  - [Get Outfit Suggestions](#3-get-outfit-suggestions)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Upload Pipeline](#upload-pipeline)
  - [Recommendation Pipeline](#recommendation-pipeline)
- [Data Model](#data-model)
- [MongoDB Atlas Vector Search Setup](#mongodb-atlas-vector-search-setup)
- [Services Overview](#services-overview)
- [Error Handling](#error-handling)
- [Development](#development)

---

## Architecture Overview

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│   Client    │     │                   FastAPI Backend                   │
│ (Web/Mobile)│────▶│                                                     │
└─────────────┘     │  ┌──────────┐  ┌─────────────┐  ┌───────────────┐   │
                    │  │  Routes  │──│  Pipelines  │──│   Services    │   │
                    │  │ (api/)   │  │ (pipelines/)│  │  (services/)  │   │
                    │  └──────────┘  └─────────────┘  └──────┬────────┘   │
                    │                                        │            │
                    └────────────────────────────────────────┼────────────┘
                                                             │
                    ┌──────────────┐  ┌───────────┐  ┌───────▼───────┐
                    │   MongoDB    │  │ Cloudinary│  │  Google Gemini│
                    │ Atlas (Vector│  │(Image CDN)│  │  (AI Analysis)│
                    │   Search)    │  └───────────┘  └───────────────┘
                    └──────────────┘
```

The system is composed of three main layers:

1. **API Layer** (`app/api/`) — FastAPI route handlers for item upload, listing, and outfit suggestions.
2. **Pipeline Layer** (`app/pipelines/`) — Orchestration logic that chains multiple services together.
3. **Service Layer** (`app/services/`) — Individual AI/ML model wrappers and external API clients.

---

## Features

### 📸 Smart Image Upload

- **Image Type Classification** — Gemini detects whether a photo is a flat product shot, mannequin display, on-model shot, or unusable.
- **Intelligent Segmentation** — Automatically selects the right segmentation method:
  - **SegFormer** (human parsing) for on-model photos — isolates the garment from the person.
  - **rembg** (background removal) for product/mannequin shots — extracts the foreground garment.
- **Cloudinary Upload** — Original, segmented cutout, and mask images are uploaded to Cloudinary with auto-generated thumbnails.

### 🧠 AI-Powered Analysis

- **FashionCLIP Embeddings** — Each item gets a 512-dimensional vector embedding for similarity search.
- **Gemini Vision Analysis** — Extracts detailed attributes:
  - Dominant colors (name, RGB, percentage)
  - Category, subcategory, fit, sleeve length, neck style
  - Pattern, fabric, texture, closure type
  - Logo/graphic presence, pockets, hood
- **Confidence Scores** — Per-attribute confidence levels from the AI model.

### 🔍 Vector Search

- **MongoDB Atlas Vector Search** — Items are searchable by visual similarity using cosine similarity on FashionCLIP embeddings.
- **Text-to-Image Search** — Natural language queries are embedded with the same FashionCLIP text encoder, enabling cross-modal search.

### 👔 AI Outfit Suggestions

- **Context-Aware Styling** — Describe an occasion (e.g., "I want to go to a party") and Gemini generates 1–2 complete outfit combinations from your wardrobe.
- **Color Coordination** — Gemini considers color harmony and style compatibility.
- **Defensive Validation** — The system verifies Gemini only references items that actually exist in the shortlist.

### 🗂️ Rich Data Model

- **16 Wardrobe Categories** — Topwear, Bottomwear, Dresswear, Drapewear, Outerwear, Innerwear, Footwear, Hosiery, Headwear, Eyewear, Earwear, Neckwear, Wristwear, Anklewear, Bags, Waistwear.
- **Category-Specific Metadata** — Each category has its own structured metadata (e.g., heel height for footwear, frame shape for eyewear).
- **Pydantic Discriminated Union** — Type-safe validation at runtime.

---

## Tech Stack

| Category          | Technology                                                         |
| ----------------- | ------------------------------------------------------------------ |
| **Framework**     | FastAPI, Uvicorn                                                   |
| **Language**      | Python 3.12+                                                       |
| **Database**      | MongoDB (with Atlas Vector Search index)                           |
| **AI / ML**       | Google Gemini 3.5 Flash, FashionCLIP (Marqo), SegFormer B2 Clothes |
| **Image Proc.**   | rembg, Pillow, NumPy, PyTorch                                      |
| **Image Hosting** | Cloudinary                                                         |
| **Package Mgr**   | uv (Python package manager)                                        |
| **Validation**    | Pydantic v2                                                        |

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point, CORS, static mount
│   ├── exceptions.py              # Custom exception classes
│   ├── api/
│   │   ├── __init__.py
│   │   ├── items.py               # POST/GET /api/items
│   │   └── suggestions.py         # POST /api/outfit-suggestions/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classification.py      # Image type & garment category classification
│   │   ├── clothing_analysis.py   # Detailed clothing attribute analysis
│   │   ├── outfit.py              # Outfit suggestion response models
│   │   └── wardrobe_schema.py     # Full wardrobe item discriminated union
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── upload_pipeline.py     # Image upload → analysis → storage
│   │   └── recommendation.py      # Query → vector search → Gemini styling
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cloudinary_client.py   # Cloudinary upload & thumbnail generation
│   │   ├── fashion_clip.py        # FashionCLIP embedding (image & text)
│   │   ├── gemini_client.py       # Google Gemini client singleton
│   │   ├── mongo_client.py        # MongoDB connection & helpers
│   │   ├── rembg_service.py       # Background removal for product shots
│   │   └── segformer.py           # Human parsing for on-model photos
│   └── static/
│       ├── index.html             # Upload demo page
│       ├── items.html             # Items listing demo page
│       └── outfits.html           # Outfit suggestions demo page
├── enums/
│   └── clothing_item.py           # Shared enums (Category, Fit, Fabric, etc.)
├── devolepment_data/              # Development/test data (not production)
├── .env.example                   # Environment variable template
├── .gitignore
├── .python-version
├── pyproject.toml                 # Project metadata & dependencies
├── uv.lock                        # Locked dependency versions
└── yolo11n-seg.pt                 # YOLO segmentation model (reserved)
```

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **uv** (Python package manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **MongoDB Atlas** cluster with Vector Search enabled
- **Cloudinary** account
- **Google Gemini API** key

### Installation

```bash
# Clone the repository
git clone https://github.com/Yashwant937363/Wardly.git
cd Wardly/backend

# Create environment file
cp .env.example .env

# Install dependencies
uv sync
```

### Environment Variables

Edit `.env` with your credentials:

```env
GEMINI_API_KEY=your_gemini_api_key
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
MONGODB_URI=your_mongodb_atlas_connection_string
MONGODB_DB=wardly
```

### Running the Server

```bash
uv run --env-file .env uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`. Static demo pages are available at:

- `http://localhost:8000/` — Upload an item
- `http://localhost:8000/items.html` — View wardrobe items
- `http://localhost:8000/outfits.html` — Get outfit suggestions

---

## API Reference

### 1. Upload a Wardrobe Item

```http
POST /api/items
```

Upload a clothing photo. The system automatically classifies the image type, segments the garment, analyzes attributes, generates an embedding, and stores everything in MongoDB.

**Request** (multipart/form-data):

| Field      | Type   | Required | Default    | Description               |
| ---------- | ------ | -------- | ---------- | ------------------------- |
| `file`     | File   | Yes      | —          | Clothing image (JPEG/PNG) |
| `owner_id` | String | No       | `user_123` | Owner identifier          |

**Response** `201 Created`:

```json
{
  "owner_id": "user_123",
  "vision_metadata": {
    "colors": {
      "dominant": [
        {"name": "Navy Blue", "rgb": [28, 45, 92], "hsv": [220, 70, 36], "hex": "#1C2D5C", "percentage": 85}
      ],
      "primary_color_family": "Blue",
      "secondary_color_family": "White"
    },
    "detected_attributes": {
      "category": "Shirt",
      "subcategory": "Casual Shirt",
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
      "category": 0.98,
      "color": 0.95,
      "fabric": 0.88,
      "pattern": 0.99,
      "fit": 0.85
    },
    "image": {
      "original_url": "https://res.cloudinary.com/.../original.jpg",
      "thumbnail_url": "https://res.cloudinary.com/.../w_300/thumbnail.jpg",
      "segmented_url": "https://res.cloudinary.com/.../cutout.png"
    },
    "segmentation": {
      "mask_url": "https://res.cloudinary.com/.../mask.png",
      "bounding_box": [120, 45, 380, 520]
    },
    "embedding": [0.012, -0.034, ...]
  }
}
```

**Error Responses**:
| Status | Description |
|--------|--------------------------------------------------|
| 422 | Image is unusable (blurry, no garment visible) |
| 400 | Unsupported garment category |
| 503 | Gemini service temporarily unavailable |

### 2. List All Items

```http
GET /api/items
```

Returns all wardrobe items for all users.

**Response** `200 OK`:

```json
[
  {
    "_id": "665a1b2c3d4e5f6a7b8c9d0e",
    "owner_id": "user_123",
    "vision_metadata": { ... }
  }
]
```

### 3. Get Outfit Suggestions

```http
POST /api/outfit-suggestions/
```

Generate outfit combinations based on a natural-language query.

**Request**:

```json
{
  "query": "I want to go to a party",
  "owner_id": "user_123"
}
```

**Response** `200 OK`:

```json
{
  "combinations": [
    {
      "title": "Party-Ready Chic",
      "items": [
        {
          "item_id": "665a1b2c3d4e5f6a7b8c9d0e",
          "category": "Topwear",
          "reason": "Black sequined top matches the party vibe",
          "image_url": "https://res.cloudinary.com/.../thumb.jpg"
        },
        {
          "item_id": "665a1b2c3d4e5f6a7b8c9d0f",
          "category": "Bottomwear",
          "reason": "High-waisted jeans balance the sparkly top",
          "image_url": "https://res.cloudinary.com/.../thumb.jpg"
        }
      ],
      "styling_notes": "Add statement earrings and heeled boots to complete the look."
    }
  ]
}
```

**Error Responses**:
| Status | Description |
|--------|--------------------------------------------------|
| 404 | No wardrobe items found for this user |
| 502 | Gemini referenced an item not in the shortlist |
| 503 | Styling service temporarily unavailable |

---

## Pipeline Deep Dive

### Upload Pipeline

The upload pipeline (`app/pipelines/upload_pipeline.py`) processes a single clothing image through these stages:

```
Upload Image
    │
    ▼
┌──────────────────────────────────────┐
│ 1. Image Classification (Gemini)     │
│    - Classifies image_type:          │
│      product_flat / product_mannequin│
│      / on_model / unusable           │
│    - Detects garment category        │
│    - Returns reason/description      │
└──────────────┬───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Segmentation Routing             │
│                                     │
│  ON_MODEL?                          │
│    ├─ Yes → SegFormer (human parse) │
│    │        Isolates garment from   │
│    │        person using cloth      │
│    │        parsing model           │
│    │                                │
│    └─ No  → rembg (background       │
│               removal)              │
│              Extracts foreground    │
│              garment from flat/     │
│              mannequin shot         │
│                                     │
│    Output: cutout PNG + mask PNG    │
│            + bounding box           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Cloudinary Upload (parallel)     │
│    - Original image                 │
│    - Segmented cutout               │
│    - Mask image                     │
│    - Thumbnail (auto-generated URL) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. Concurrent Analysis              │
│                                     │
│    ┌─── Gemini Vision ────┐         │
│    │  - Dominant colors   │         │
│    │  - Attributes        │         │
│    │  - Confidence scores │         │
│    └──────────────────────┘         │
│                                     │
│    ┌─── FashionCLIP ─────┐          │
│    │  - 512-dim embedding │         │
│    └──────────────────────┘         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. Build Document & Save to MongoDB │
└─────────────────────────────────────┘
```

### Recommendation Pipeline

The recommendation pipeline (`app/pipelines/recommendation.py`) generates outfit suggestions:

```
User Query (e.g., "I want to go to a party")
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Text Embedding (FashionCLIP)     │
│    - Encodes query into 512-dim     │
│      vector in the same space as    │
│      stored image embeddings        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. MongoDB Atlas $vectorSearch      │
│    - Cosine similarity search       │
│    - Filtered by owner_id           │
│    - Returns top 15 candidate items │
│    - Includes thumbnail URLs        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Gemini Outfit Generation         │
│    - Receives shortlisted items     │
│    - Builds 1-2 outfit combos       │
│    - Considers:                     │
│      • Occasion/style match         │
│      • Color coordination           │
│      • Category completeness        │
│    - Returns structured JSON        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. Defensive Validation             │
│    - Verifies all item_ids exist    │
│      in the shortlist               │
│    - Fetches missing thumbnail URLs │
│    - Raises ValueError if Gemini    │
│      hallucinated items             │
└──────────────┬──────────────────────┘
               │
               ▼
         Return OutfitSuggestions
```

---

## Data Model

### Wardrobe Item (MongoDB Document)

```json
{
  "_id": ObjectId("..."),
  "owner_id": "user_123",
  "vision_metadata": {
    "colors": {
      "dominant": [
        {
          "name": "Navy Blue",
          "rgb": [28, 45, 92],
          "hsv": [220, 70, 36],
          "hex": "#1C2D5C",
          "percentage": 85
        }
      ],
      "primary_color_family": "Blue",
      "secondary_color_family": "White"
    },
    "detected_attributes": {
      "category": "Shirt",
      "subcategory": "Casual Shirt",
      "fit": "Regular Fit",
      "sleeve_length": "Full Sleeve",
      "neck_style": "Button Down",
      "pattern": "Solid",
      "fabric": "Cotton",
      "texture": "Smooth",
      "closure": "Buttons",
      "neckline": null,
      "logo_present": false,
      "graphic_present": false,
      "hood": false,
      "pockets": 2
    },
    "confidence": {
      "category": 0.98,
      "color": 0.95,
      "fabric": 0.88,
      "pattern": 0.99,
      "fit": 0.85
    },
    "image": {
      "original_url": "https://...",
      "thumbnail_url": "https://...",
      "segmented_url": "https://..."
    },
    "segmentation": {
      "mask_url": "https://...",
      "bounding_box": [120, 45, 380, 520]
    },
    "embedding": [0.012, -0.034, ...]
  }
}
```

### Category-Specific Schema (Pydantic Discriminated Union)

The `wardrobe_schema.py` defines a discriminated union with 16 categories, each with its own metadata shape:

| Category   | Metadata Fields                                   |
| ---------- | ------------------------------------------------- |
| Topwear    | sleeveType, neckline, fit, pattern, length        |
| Bottomwear | fit, length, waistRise, closure                   |
| Dresswear  | length, neckline, sleeveType, fit                 |
| Drapewear  | drapeType, blouseIncluded, workType, fabricWeight |
| Outerwear  | closureType, warmthLevel                          |
| Innerwear  | (none)                                            |
| Footwear   | heelHeight, closureType, soleType                 |
| Hosiery    | length, thickness                                 |
| Headwear   | (none)                                            |
| Eyewear    | frameShape, lensType                              |
| Earwear    | closure                                           |
| Neckwear   | length                                            |
| Wristwear  | strapMaterial                                     |
| Anklewear  | hasBells                                          |
| Bags       | strapType, compartments                           |
| Waistwear  | buckleType, adjustable                            |

---

## MongoDB Atlas Vector Search Setup

To enable vector search, create a search index on the `wardrobe_items` collection:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "vision_metadata.embedding": {
        "type": "knnVector",
        "dimensions": 512,
        "similarity": "cosine"
      },
      "owner_id": {
        "type": "filter"
      }
    }
  }
}
```

**Index name**: `wardrobe_vector_index`  
**Collection**: `wardrobe_items`  
**Database**: `wardly`

---

## Services Overview

### `gemini_client.py`

Singleton Google Gemini client initialized with the API key from environment variables. Used for:

- Image type classification
- Garment category detection
- Detailed clothing attribute analysis
- Outfit suggestion generation

### `fashion_clip.py`

Wraps Marqo's FashionCLIP model loaded via `open_clip`:

- `generate_embedding(image_path)` — 512-dim image embedding
- `embed_query(text)` — 512-dim text embedding in the same latent space
- Lazy-loaded singleton pattern (model loads on first use)

### `segformer.py`

Human parsing using `mattmdjaga/segformer_b2_clothes`:

- Maps garment categories to SegFormer label names (Upper-clothes, Pants, Skirt, Dress, etc.)
- Generates transparent PNG cutout + mask + bounding box
- Raises `UnsupportedCategoryError` for unmapped categories

### `rembg_service.py`

Background removal for product/mannequin shots using `rembg`:

- Extracts the full foreground as a transparent PNG
- Generates alpha mask and bounding box
- Raises `UnusableImageError` if no foreground detected

### `cloudinary_client.py`

Cloudinary integration:

- `upload_to_cloudinary(file_path, folder)` — uploads and returns secure URL
- `build_thumbnail_url(original_url, width)` — generates on-the-fly thumbnail URL via Cloudinary transformations

### `mongo_client.py`

MongoDB connection management:

- Thread-safe singleton `MongoClient` (connection pooling)
- `save_wardrobe_item(document)` — inserts a document
- `_mongo_safe(value)` — recursively converts ObjectId to string for JSON serialization

---

## Error Handling

Custom exceptions defined in `app/exceptions.py`:

| Exception                  | HTTP Status | Description                                           |
| -------------------------- | ----------- | ----------------------------------------------------- |
| `UnusableImageError`       | 422         | Image is too blurry, cropped, or has no clear garment |
| `UnsupportedCategoryError` | 400         | Garment category has no SegFormer label mapping       |

API routes handle these with appropriate HTTP status codes and descriptive error messages. Gemini `ServerError` exceptions are caught and returned as 503 (Service Unavailable).

---

## Development

### Adding a New Service

1. Create the service file in `app/services/`
2. Implement the logic with lazy-loaded singleton pattern (if using ML models)
3. Import and use in the appropriate pipeline

### Adding a New Category

1. Add the category to `Category` enum in `app/models/wardrobe_schema.py`
2. Create a metadata model class
3. Add a new item subclass with the `Literal[Category.NEW_CATEGORY]` discriminator
4. Add the item class to the `WardrobeItem` union type
5. Update the SegFormer label mapping in `app/services/segformer.py` if needed

### Running Tests

```bash
# (Tests not yet implemented — run the pipeline directly)
uv run python -m app.pipelines.upload_pipeline
uv run python -m app.pipelines.recommendation
```

### Code Style

The project follows standard Python conventions. Key patterns:

- **Lazy-loaded singletons** for ML models (loaded on first use, cached globally)
- **`asyncio.to_thread`** for blocking operations (ML inference, network calls)
- **Pydantic v2** for all data validation and serialization
- **Structured output** with Gemini's `response_format` schema for reliable JSON parsing
