from pydantic import BaseModel
from typing import List, Optional


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    source_title: str
    source_url: str
    related_question: str

    model_config = {
        "from_attributes": True
    }


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int