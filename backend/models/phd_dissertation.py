from pydantic import BaseModel
from typing import List, Optional

class Title(BaseModel):
    en: str
    sr: str

class Details(BaseModel):
    en: str
    sr: str

class PhdDissertation(BaseModel):
    _id: str
    title: Title
    details: Details