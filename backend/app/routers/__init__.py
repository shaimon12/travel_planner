from fastapi import APIRouter
from app.routers.trips import router as trips_router
from app.routers.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(trips_router)

__all__ = ["api_router"]

