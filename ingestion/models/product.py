from typing import List, Dict, Optional
from pydantic import BaseModel
from .product_detail import ProductDetail

class Product(BaseModel):
    _id: str
    pid: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    selling_price: Optional[str] = None
    actual_price: Optional[str] = None
    discount: Optional[str] = None
    average_rating: Optional[str] = None
    out_of_stock: Optional[bool] = False
    seller: Optional[str] = None
    url: Optional[str] = None
    images: Optional[List[str]] = []
    product_details: Optional[List[ProductDetail]] = []

    @staticmethod
    def from_raw(data: Dict):
        details = [ProductDetail(key=k, value=v) for d in data.get('product_details', []) for k, v in d.items()]
        return Product(
            _id=data['_id'],
            pid=data.get('pid'),
            title=data.get('title'),
            description=data.get('description'),
            brand=data.get('brand'),
            category=data.get('category'),
            sub_category=data.get('sub_category'),
            selling_price=data.get('selling_price'),
            actual_price=data.get('actual_price'),
            discount=data.get('discount'),
            average_rating=data.get('average_rating'),
            out_of_stock=data.get('out_of_stock', False),
            seller=data.get('seller'),
            url=data.get('url'),
            images=data.get('images', []),
            product_details=details
        )