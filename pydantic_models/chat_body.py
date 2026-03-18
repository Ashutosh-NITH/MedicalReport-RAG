from pydantic import BaseModel
from typing import Optional


class ChatBody(BaseModel):
    query: Optional[str] = None