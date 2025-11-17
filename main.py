import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from bson import ObjectId
from datetime import datetime

from database import db, create_document, get_documents
from schemas import User as UserSchema, WardrobeItem as WardrobeItemSchema, OutfitRecommendation as OutfitRecommendationSchema

# OpenAI SDK
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Utilities ---------------------

def oid_str(o: Any) -> str:
    try:
        return str(o)
    except Exception:
        return o


def to_serializable(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = dict(doc)
    if d.get("_id"):
        d["_id"] = oid_str(d["_id"])
    # Convert datetimes
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# --------------------- Models ---------------------

class UpsertUserRequest(BaseModel):
    email: str
    name: Optional[str] = None
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=300)
    body_type: Optional[str] = None
    style_preferences: Optional[List[str]] = None


class AnalyzeWardrobeRequest(BaseModel):
    user_id: str
    image_url: HttpUrl


class CreateWardrobeRequest(AnalyzeWardrobeRequest):
    # Optional manual overrides
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    style: Optional[str] = None
    details: Optional[str] = None
    season: Optional[List[str]] = None
    brand: Optional[str] = None
    notes: Optional[str] = None


class GenerateRecommendationRequest(BaseModel):
    user_id: str
    occasion: Optional[str] = None
    weather: Optional[str] = None


class QuickSuggestWithoutWardrobeRequest(BaseModel):
    height_cm: float
    weight_kg: float
    body_type: Optional[str] = None
    style_choice: Optional[str] = Field(None, description="e.g., classic, streetwear, minimalist")
    desired_category: Optional[str] = Field(None, description="If user wants to pick a type, like 'smart casual', 'evening', 'business' ")


# --------------------- Routes ---------------------

@app.get("/")
def read_root():
    return {"message": "Style Sage API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:80]}"
    return response


# ---- Users ----
@app.post("/users")
def upsert_user(payload: UpsertUserRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    existing = db["user"].find_one({"email": payload.email})
    data = payload.model_dump()
    data["updated_at"] = datetime.utcnow()

    if existing:
        db["user"].update_one({"_id": existing["_id"]}, {"$set": data})
        user = db["user"].find_one({"_id": existing["_id"]})
    else:
        data["created_at"] = datetime.utcnow()
        inserted = db["user"].insert_one(data)
        user = db["user"].find_one({"_id": inserted.inserted_id})

    return to_serializable(user)


@app.get("/users/{user_id}")
def get_user(user_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    try:
        user = db["user"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return to_serializable(user)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")


# ---- Wardrobe ----
@app.post("/wardrobe/analyze")
def analyze_and_create_item(payload: CreateWardrobeRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured. Set OPENAI_API_KEY.")

    system_prompt = (
        "You are an expert fashion stylist. Analyze the clothing image and respond with a JSON object "
        "with keys: category, subcategory, color, material, style, details, season (array if inferable), brand (if visible). "
        "Focus only on the visible item, be specific and use refined color names (e.g., 'Navy Suede')."
    )
    user_content = [
        {"type": "text", "text": "Analyze this clothing item and return a single JSON object only."},
        {"type": "image_url", "image_url": {"url": str(payload.image_url)}},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:120]}")

    try:
        import json
        attrs = json.loads(content or "{}")
    except Exception:
        attrs = {}

    data = WardrobeItemSchema(
        user_id=payload.user_id,
        image_url=str(payload.image_url),
        category=payload.category or attrs.get("category") or "Item",
        subcategory=payload.subcategory or attrs.get("subcategory"),
        color=payload.color or attrs.get("color"),
        material=payload.material or attrs.get("material"),
        style=payload.style or attrs.get("style"),
        details=payload.details or attrs.get("details"),
        season=payload.season or attrs.get("season"),
        brand=payload.brand or attrs.get("brand"),
        notes=payload.notes,
    )

    item_id = create_document("wardrobeitem", data)
    item = db["wardrobeitem"].find_one({"_id": ObjectId(item_id)})
    return to_serializable(item)


@app.get("/users/{user_id}/wardrobe")
def list_wardrobe(user_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    items = db["wardrobeitem"].find({"user_id": user_id}).sort("created_at", -1)
    return [to_serializable(i) for i in items]


# ---- Recommendations ----
@app.post("/recommendations")
def generate_recommendation(payload: GenerateRecommendationRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured. Set OPENAI_API_KEY.")

    user = db["user"].find_one({"_id": ObjectId(payload.user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    wardrobe = list(db["wardrobeitem"].find({"user_id": payload.user_id}))
    if not wardrobe:
        raise HTTPException(status_code=400, detail="No wardrobe items. Add items or use the without-wardrobe flow.")

    inventory = []
    for it in wardrobe:
        inventory.append({
            "_id": oid_str(it.get("_id")),
            "category": it.get("category"),
            "subcategory": it.get("subcategory"),
            "color": it.get("color"),
            "material": it.get("material"),
            "style": it.get("style"),
            "details": it.get("details"),
            "brand": it.get("brand"),
            "image_url": it.get("image_url"),
        })

    import json
    system = (
        "You are a senior fashion stylist. Select a cohesive outfit based ONLY on the provided wardrobe inventory. "
        "Return a strict JSON object with fields: outfit_name, justification, items (array). Each items[] must have: "
        "wardrobe_item_id (from inventory _id), category, style (if known), color (specific), and specific_name (e.g., 'Navy Suede Loafers'). "
        "Consider user height, weight, body type, preferences, occasion, and weather."
    )
    user_msg = {
        "profile": {
            "height_cm": user.get("height_cm"),
            "weight_kg": user.get("weight_kg"),
            "body_type": user.get("body_type"),
            "style_preferences": user.get("style_preferences"),
        },
        "occasion": payload.occasion,
        "weather": payload.weather,
        "inventory": inventory,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:160]}")

    # Validate and persist recommendation
    items = []
    for it in data.get("items", []):
        items.append({
            "wardrobe_item_id": it.get("wardrobe_item_id"),
            "category": it.get("category"),
            "style": it.get("style"),
            "color": it.get("color"),
            "specific_name": it.get("specific_name"),
        })

    rec = OutfitRecommendationSchema(
        user_id=payload.user_id,
        outfit_name=data.get("outfit_name", "Curated Look"),
        justification=data.get("justification", ""),
        occasion=payload.occasion,
        weather=payload.weather,
        items=items,
    )
    rec_id = create_document("outfitrecommendation", rec)
    saved = db["outfitrecommendation"].find_one({"_id": ObjectId(rec_id)})

    # Attach linked images for convenience
    for it in data.get("items", []):
        match = db["wardrobeitem"].find_one({"_id": ObjectId(it.get("wardrobe_item_id"))}) if ObjectId.is_valid(it.get("wardrobe_item_id", "")) else None
        it["image_url"] = match.get("image_url") if match else None

    return {
        "_id": rec_id,
        "outfit_name": data.get("outfit_name"),
        "justification": data.get("justification"),
        "items": data.get("items", []),
    }


@app.post("/recommendations/without-wardrobe")
def quick_suggest(payload: QuickSuggestWithoutWardrobeRequest):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI not configured. Set OPENAI_API_KEY.")

    system = (
        "You are a luxury fashion stylist. Recommend a cohesive outfit with SPECIFIC item names (e.g., 'Navy Suede Loafers', "
        "'Cream Merino Turtleneck', 'Charcoal Flannel Trousers'). Return JSON with fields: outfit_name, justification, "
        "items: array of {category, specific_name, color, why}. Avoid generic words like 'black shoes'; be precise and premium. "
        "No images."
    )
    import json
    user_msg = payload.model_dump()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_msg)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)[:160]}")


@app.post("/favorites/{recommendation_id}")
def toggle_favorite(recommendation_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    if not ObjectId.is_valid(recommendation_id):
        raise HTTPException(status_code=400, detail="Invalid id")
    rec = db["outfitrecommendation"].find_one({"_id": ObjectId(recommendation_id)})
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    new_val = not rec.get("is_favorite", False)
    db["outfitrecommendation"].update_one({"_id": rec["_id"]}, {"$set": {"is_favorite": new_val, "updated_at": datetime.utcnow()}})
    return {"_id": recommendation_id, "is_favorite": new_val}
