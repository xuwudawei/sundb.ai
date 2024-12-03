import logging

from fastapi import APIRouter, Depends, Query
from fastapi_pagination import Params, Page

from app.rag.knowledge_base.index_store import init_kb_tidb_vector_store, init_kb_tidb_graph_store
from app.repositories.chunk import ChunkRepo
from app.repositories.embedding_model import embed_model_repo
from app.repositories.llm import llm_repo

from .models import (
    KnowledgeBaseDetail,
    KnowledgeBaseItem,
    KnowledgeBaseCreate, KnowledgeBaseUpdate, VectorIndexError, KGIndexError
)
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.exceptions import (
    InternalServerError,
    KnowledgeBaseNotFoundError,
    KBNoVectorIndexConfiguredError
)
from app.models import (
    KnowledgeBase,
)
from app.models.data_source import DataSource
from app.tasks import (
    build_kg_index_for_chunk,
    build_index_for_document,
)
from app.repositories import knowledge_base_repo, data_source_repo, document_repo
from app.tasks.knowledge_base import (
    import_documents_for_knowledge_base,
    stats_for_knowledge_base,
    purge_knowledge_base_related_resources
)
from ..document.models import DocumentItem, DocumentFilters

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/admin/knowledge_bases")
def create_knowledge_base(
    session: SessionDep,
    user: CurrentSuperuserDep,
    create: KnowledgeBaseCreate
) -> KnowledgeBaseDetail:
    try:
        data_sources = [
            data_source_repo.create(session, DataSource(
                name=data_source.name,
                description='',
                user_id=user.id,
                data_source_type=data_source.data_source_type,
                config=data_source.config,
            )) for data_source in create.data_sources
        ]

        if not create.llm_id:
            create.llm_id = llm_repo.must_get_default_llm(session).id

        if not create.embedding_model_id:
            create.embedding_model_id = embed_model_repo.must_get_default_model(session).id

        knowledge_base = KnowledgeBase(
            name=create.name,
            description=create.description,
            index_methods=create.index_methods,
            llm_id=create.llm_id,
            embedding_model_id=create.embedding_model_id,
            data_sources=data_sources,
            created_by=user.id,
            updated_by=user.id,
        )
        knowledge_base = knowledge_base_repo.create(session, knowledge_base)

        # Ensure the knowledge-base corresponding table schema are initialized.
        init_kb_tidb_vector_store(session, knowledge_base)
        init_kb_tidb_graph_store(session, knowledge_base)

        # Trigger import and index documents for knowledge base
        import_documents_for_knowledge_base.delay(knowledge_base.id)

        return knowledge_base
    except KBNoVectorIndexConfiguredError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases")
def list_knowledge_bases(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: Params = Depends(),
) -> Page[KnowledgeBaseItem]:
    return knowledge_base_repo.paginate(session, params)


@router.get("/admin/knowledge_bases/{knowledge_base_id}")
def get_knowledge_base(
    session: SessionDep,
    user: CurrentSuperuserDep,
    knowledge_base_id: int,
) -> KnowledgeBaseDetail:
    try:
        return knowledge_base_repo.must_get(session, knowledge_base_id)
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.put("/admin/knowledge_bases/{knowledge_base_id}")
def update_knowledge_base_setting(
    session: SessionDep,
    user: CurrentSuperuserDep,
    knowledge_base_id: int,
    update: KnowledgeBaseUpdate
) -> KnowledgeBaseDetail:
    try:
        knowledge_base = knowledge_base_repo.must_get(session, knowledge_base_id)
        knowledge_base = knowledge_base_repo.update(session, knowledge_base, update)
        return knowledge_base
    except KnowledgeBaseNotFoundError as e:
        raise e
    except KBNoVectorIndexConfiguredError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.delete("/admin/knowledge_bases/{knowledge_base_id}")
def delete_knowledge_base(
    session: SessionDep,
    user: CurrentSuperuserDep,
    knowledge_base_id: int
):
    try:
        knowledge_base = knowledge_base_repo.must_get(session, knowledge_base_id)
        knowledge_base_repo.delete(session, knowledge_base)

        # Trigger purge knowledge base related resources after 5 seconds.
        purge_knowledge_base_related_resources.apply_async(
            args=[knowledge_base_id],
            countdown=5
        )

        return {
            "detail": f"Knowledge base #{knowledge_base_id} is deleted successfully"
        }
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{knowledge_base_id}/overview")
def get_knowledge_base_index_overview(
    session: SessionDep,
    user: CurrentSuperuserDep,
    knowledge_base_id: int,
) -> dict:
    try:
        knowledge_base = knowledge_base_repo.must_get(session, knowledge_base_id)

        stats_for_knowledge_base.delay(knowledge_base.id)

        return knowledge_base_repo.get_index_overview(session, knowledge_base)
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/vector-index-errors")
def list_kb_vector_index_errors(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    params: Params = Depends(),
) -> Page[VectorIndexError]:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        return knowledge_base_repo.list_vector_index_built_errors(session, kb, params)
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.get("/admin/knowledge_bases/{kb_id}/kg-index-errors")
def list_kb_kg_index_errors(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
    params: Params = Depends(),
) -> Page[KGIndexError]:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)
        return knowledge_base_repo.list_kg_index_built_errors(session, kb, params)
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()


@router.post("/admin/knowledge_bases/{kb_id}/retry-failed-index-tasks")
def retry_failed_tasks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    kb_id: int,
) -> dict:
    try:
        kb = knowledge_base_repo.must_get(session, kb_id)

        # Retry failed vector index tasks.
        document_ids = knowledge_base_repo.set_failed_documents_status_to_pending(session, kb)
        for document_id in document_ids:
            build_index_for_document.delay(kb_id, document_id)
        logger.info(f"Triggered {len(document_ids)} documents to rebuilt vector index." )

        # Retry failed kg index tasks.
        chunk_ids = knowledge_base_repo.set_failed_chunks_status_to_pending(session, kb)
        for chunk_id in chunk_ids:
            build_kg_index_for_chunk.delay(kb_id, chunk_id)
        logger.info(f"Triggered {len(chunk_ids)} chunks to rebuilt knowledge graph index." )

        return {
            "detail": f"Triggered reindex {len(document_ids)} documents and {len(chunk_ids)} chunks of knowledge base #{kb_id}."
        }
    except KnowledgeBaseNotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(e)
        raise InternalServerError()
