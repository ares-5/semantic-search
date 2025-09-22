from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    _id: str
    title: str
    description: str
    brand: str
    category: str
    seller: str
    actual_price: Optional[str] = None
    discount: Optional[str] = None
    images: Optional[List[str]] = None
    url: Optional[str] = None
