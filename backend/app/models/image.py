from typing import Optional
from datetime import datetime
from sqlmodel import Field, Column, Text, DateTime, JSON
from .base import UpdatableBaseModel

class Image(UpdatableBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(max_length=512)
    caption: str = Field(sa_column=Column(Text, nullable=True))
    text_snippets: str = Field(sa_column=Column(Text, nullable=True))
    description: str = Field(sa_column=Column(Text, nullable=True))
    source_document_id: int = Field(foreign_key="documents.id")
    meta: dict = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)

    __tablename__ = "images"