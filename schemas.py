"""
Database Schemas for Style Sage

Each Pydantic model represents a MongoDB collection (collection name = class name lowercased).
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal


class User(BaseModel):
    email: str = Field(..., description="User email (unique)")
    name: Optional[str] = Field(None, description="Full name")
    password_hash: Optional[str] = Field(None, description="Hashed password (if using email/password)")
    auth_provider: Literal["password", "google", "apple", "facebook"] = "password"
    avatar_url: Optional[HttpUrl] = None
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=300)
    body_type: Optional[str] = Field(None, description="e.g., rectangle, triangle, inverted triangle, hourglass")
    style_preferences: Optional[List[str]] = Field(default=None, description="e.g., classic, streetwear, minimalist, smart casual")


class WardrobeItem(BaseModel):
    user_id: str = Field(..., description="Owner user id (stringified ObjectId)")
    image_url: HttpUrl = Field(..., description="Publicly accessible image URL")
    category: str = Field(..., description="e.g., Sweater, Pants, Shoes")
    subcategory: Optional[str] = Field(None)
    color: Optional[str] = Field(None)
    material: Optional[str] = Field(None)
    style: Optional[str] = Field(None)
    details: Optional[str] = Field(None)
    season: Optional[List[str]] = Field(default=None, description="e.g., [winter, fall]")
    brand: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)


class OutfitItemRef(BaseModel):
    wardrobe_item_id: str = Field(..., description="Stringified ObjectId of WardrobeItem")
    category: str
    style: Optional[str] = None
    color: Optional[str] = None
    specific_name: Optional[str] = Field(None, description="Precise item naming like 'Navy Suede Loafers'")


class OutfitRecommendation(BaseModel):
    user_id: str
    outfit_name: str
    justification: str
    occasion: Optional[str] = None
    weather: Optional[str] = None
    items: List[OutfitItemRef]
    rating: Optional[float] = Field(None, ge=0, le=5)
    is_favorite: bool = False
