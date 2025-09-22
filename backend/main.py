from fastapi import FastAPI
from routes import products, search

app = FastAPI(title="Semantic Search Api")

app.include_router(products.router, prefix="/products", tags=["products"])
app.include_router(search.router, prefix="/search", tags=["search"])