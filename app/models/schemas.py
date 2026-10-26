from typing import Optional, List

from pydantic import BaseModel

# --- 2. 数据模型 ---
class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    thread_id: str
    # 前端未传 user_id（静态包只发送 message/image_url/thread_id），给默认值避免 422
    user_id: str = "default"