from fastapi import FastAPI
from routes import phd_dissertations
from routes import search
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Semantic Search Api")

origins = ["http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(phd_dissertations.router, prefix="/dissertations", tags=["dissertations"])
app.include_router(search.router, prefix="/search", tags=["search"])