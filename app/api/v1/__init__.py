from fastapi import APIRouter

from .files import router as files_router

v1_router = APIRouter()

v1_router.include_router(files_router)