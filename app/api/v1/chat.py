from fastapi import APIRouter
from app.models.schemas import ChatRequest
from app.agents.graph_chef import search_recipes,clear_messages,getmessages
from fastapi.responses import StreamingResponse
router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话"""
    return StreamingResponse(search_recipes(request.message,request.image_url,request.thread_id,request.user_id),
                             media_type="text/event-stream")


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages = getmessages(thread_id)
    return {"messages":messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    clear_messages(thread_id)
    return {"success":True}