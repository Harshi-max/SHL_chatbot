from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.rag.catalog_store import CatalogStore
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.ui import router as ui_router
from app.utils.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger = get_logger(__name__)
    store = CatalogStore()
    store.load()
    logger.info("Catalog loaded with %d assessments", store.count)
    yield


app = FastAPI(
    title="SHL Conversational Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ui_router)
