from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, Recommendation, RecommendRequest, RecommendResponse
from app.rag.catalog_store import CatalogStore
from app.rag.retriever import HybridRetriever
from app.services.agent import RecommendationAgent
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        store = CatalogStore()
        if store.count == 0 and not store.records:
            store.load()
        agent = RecommendationAgent(store)
        result = agent.handle([m.model_dump() for m in request.messages])

        recs = [Recommendation.model_validate(item) for item in result.recommendations]
        return ChatResponse(
            reply=result.reply,
            recommendations=recs,
            end_of_conversation=result.end_of_conversation,
        )
    except Exception as exc:
        logger.exception("Failed /chat request")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    try:
        store = CatalogStore()
        if store.count == 0 and not store.records:
            store.load()
        retriever = HybridRetriever(store)
        hits = retriever.retrieve(request.query, top_k=5)
        
        recs = [Recommendation.model_validate(item) for item in hits]
        return RecommendResponse(recommendations=recs)
    except Exception as exc:
        logger.exception("Failed /recommend request")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
