from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.utils.logging import get_logger

router = APIRouter(include_in_schema=False)
logger = get_logger(__name__)

# Initialize templates with proper string path
try:
    templates = Jinja2Templates(directory="app/templates")
except Exception as e:
    logger.error(f"Failed to initialize Jinja2Templates with directory 'app/templates': {e}")
    templates = None


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    if not templates:
        logger.error("Template engine not initialized")
        return HTMLResponse("<h1>SHL Recommender</h1><p>System configuration error: Templates missing.</p>")
    try:
        # Modern TemplateResponse signature (request as first positional or keyword arg)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request}
        )
    except Exception as e:
        logger.exception(f"Exception rendering index.html: {str(e)}")
        # Safe fallback response
        return HTMLResponse("<h1>SHL Recommender</h1><p>Internal Server Error during rendering.</p>")


@router.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui(request: Request) -> HTMLResponse:
    if not templates:
        logger.error("Template engine not initialized")
        return HTMLResponse("<h1>SHL Chat</h1><p>System configuration error: Templates missing.</p>")
    try:
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={"request": request}
        )
    except Exception as e:
        logger.exception(f"Exception rendering chat.html: {str(e)}")
        # Safe fallback response
        return HTMLResponse("<h1>SHL Chat</h1><p>Internal Server Error during rendering.</p>")
