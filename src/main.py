from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import inventory_manager_v1, inventory_manager_v2, recommendation
from src.config import settings

# Initialize settings
settings = settings.Settings()

app = FastAPI(title="Application Management API", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(
    inventory_manager_v1.router, prefix="/api", tags=["inventory_manager_v1"]
)
app.include_router(
    inventory_manager_v2.router, prefix="/api", tags=["inventory_manager_v2"]
)
app.include_router(recommendation.router, prefix="/api", tags=["recommendation"])
