from typing import Optional, List

from fastapi import APIRouter
from app.models import Document
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.retrieve import RetrieveService
from llama_index.core.schema import NodeWithScore

router = APIRouter()


@router.get("/admin/retrieve/documents")
async def retrieve_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    top_k: Optional[int] = 5,
) -> List[Document]:
    retrieve_service = RetrieveService(session, chat_engine)
    return retrieve_service.retrieve(question, top_k=top_k)


@router.get("/admin/embedding_retrieve")
async def embedding_retrieve(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    top_k: Optional[int] = 5,
) -> List[NodeWithScore]:
    retrieve_service = RetrieveService(session, chat_engine)
    return retrieve_service._embedding_retrieve(question, top_k=top_k)
