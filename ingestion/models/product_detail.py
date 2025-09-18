from pydantic import BaseModel

class ProductDetail(BaseModel):
    key: str
    value: str
