import traceback
from uuid import UUID
from sqlmodel import Session, select
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
from llama_index.core.llms.llm import LLM
import logging
from app.celery import app as celery_app
from app.core.db import engine
from app.models import (
    Document as DBDocument,
    Chunk as DBChunk,
    DataSource,
    DocIndexTaskStatus,
    KgIndexStatus,
)
from app.rag.build import BuildService
from app.rag.chat_config import get_llm, get_default_llm
from app.utils.dspy import get_dspy_lm_by_llama_llm
from app.repositories import data_source_repo

logger = get_task_logger(__name__)
logger.setLevel(logging.DEBUG)


def get_llm_by_data_source(session: Session, data_source: DataSource) -> LLM:
    if data_source.llm_id is None:
        return get_default_llm(session)

    return get_llm(
        provider=data_source.llm.provider,
        model=data_source.llm.model,
        config=data_source.llm.config,
        credentials=data_source.llm.credentials,
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def build_vector_index_from_document(self, data_source_id: int, document_id: int):
    try:
        print("\nNew process: Trying to build_vector_index_from_document")
        with Session(engine, expire_on_commit=False) as session:
            data_source = data_source_repo.get(session, data_source_id)
            print(f"Received data source: {data_source}")
            if data_source is None:
                logger.error(f"Data source with id {data_source_id} not found")
                return

            db_document = session.get(DBDocument, document_id)
            if db_document is None:
                logger.error(f"Document {document_id} not found")
                return

            # Include FAILED state to allow reprocessing
            if db_document.index_status not in (
                DocIndexTaskStatus.PENDING,
                DocIndexTaskStatus.NOT_STARTED,
                DocIndexTaskStatus.RUNNING,
                DocIndexTaskStatus.FAILED,
            ):
                logger.info(f"Document {document_id} not in a processable state")
                return

            try:
                llm = get_llm_by_data_source(session, data_source)
            except ValueError as e:
                if self.request.retries >= self.max_retries:
                    # Retries exhausted
                    error_msg = traceback.format_exc()
                    logger.error(f"Max retries exceeded while getting LLM for document {document_id}: {error_msg}")
                    db_document.index_status = DocIndexTaskStatus.FAILED
                    db_document.index_result = error_msg
                    session.add(db_document)
                    session.commit()
                    raise MaxRetriesExceededError(f"Max retries exceeded: {e}")
                else:
                    logger.warning(
                        f"Error while getting LLM for document {document_id}: {e}, task will be retried after 1 minute"
                    )
                    raise self.retry(countdown=60, exc=e)

            db_document.index_status = DocIndexTaskStatus.RUNNING
            session.add(db_document)
            session.commit()

            if (
                session.exec(
                    select(DBChunk).where(DBChunk.document_id == document_id)
                ).first()
                is not None
            ):
                logger.info(f"Document {document_id} already indexed")
                return

        # Proceed with indexing
        with Session(engine) as index_session:
            build_service = BuildService(
                llm=llm,
                dspy_lm=get_dspy_lm_by_llama_llm(llm),
            )
            build_service.build_vector_index_from_document(index_session, db_document)

        # After successful indexing
        with Session(engine, expire_on_commit=False) as session:
            db_document.index_status = DocIndexTaskStatus.COMPLETED
            session.add(db_document)
            session.commit()
            logger.info(f"Document {document_id} indexed successfully")

            data_source = data_source_repo.get(session, data_source_id)
            if data_source and data_source.build_kg_index:
                for chunk in session.exec(
                    select(DBChunk).where(DBChunk.document_id == document_id)
                ):
                    build_kg_index_from_chunk.delay(data_source_id, document_id, chunk.id)

    except Exception as e:
        with Session(engine, expire_on_commit=False) as session:
            db_document = session.get(DBDocument, document_id)
            if self.request.retries >= self.max_retries:
                # Retries exhausted
                error_msg = traceback.format_exc()
                logger.error(f"Max retries exceeded while indexing document {document_id}: {error_msg}")
                if db_document:
                    db_document.index_status = DocIndexTaskStatus.FAILED
                    db_document.index_result = error_msg
                    session.add(db_document)
                    session.commit()
                raise MaxRetriesExceededError(f"Max retries exceeded: {e}")
            else:
                # Retry the task
                logger.error(f"Error while indexing document {document_id}: {e}, retrying...")
                session.rollback()  # Rollback any changes
                if db_document and db_document.index_status != DocIndexTaskStatus.RUNNING:
                    db_document.index_status = DocIndexTaskStatus.PENDING
                    session.add(db_document)
                    session.commit()
                raise self.retry(exc=e)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def build_kg_index_from_chunk(self, data_source_id: int, document_id: int, chunk_id: UUID):
    try:
        with Session(engine, expire_on_commit=False) as session:
            data_source = data_source_repo.get(session, data_source_id)
            if data_source is None:
                logger.error(f"Data source with id {data_source_id} not found")
                return

            db_chunk = session.get(DBChunk, chunk_id)
            if db_chunk is None:
                logger.error(f"Chunk {chunk_id} not found")
                return

            # Include FAILED state to allow reprocessing
            if db_chunk.index_status not in (
                KgIndexStatus.PENDING,
                KgIndexStatus.NOT_STARTED,
                KgIndexStatus.RUNNING,
                KgIndexStatus.FAILED,
            ):
                logger.info(f"Chunk {chunk_id} not in a processable state")
                return

            try:
                llm = get_llm_by_data_source(session, data_source)
            except ValueError as e:
                if self.request.retries >= self.max_retries:
                    # Retries exhausted
                    error_msg = traceback.format_exc()
                    logger.error(f"Max retries exceeded while getting LLM for chunk {chunk_id}: {error_msg}")
                    db_chunk.index_status = KgIndexStatus.FAILED
                    db_chunk.index_result = error_msg
                    session.add(db_chunk)
                    session.commit()
                    raise MaxRetriesExceededError(f"Max retries exceeded: {e}")
                else:
                    logger.warning(
                        f"Error while getting LLM for chunk {chunk_id}: {e}, task will be retried after 1 minute"
                    )
                    raise self.retry(countdown=60, exc=e)

            db_chunk.index_status = KgIndexStatus.RUNNING
            session.add(db_chunk)
            session.commit()

        # Proceed with indexing
        with Session(engine) as index_session:
            build_service = BuildService(
                llm=llm,
                dspy_lm=get_dspy_lm_by_llama_llm(llm),
            )
            build_service.build_kg_index_from_chunk(index_session, db_chunk)

        # After successful indexing
        with Session(engine, expire_on_commit=False) as session:
            db_chunk.index_status = KgIndexStatus.COMPLETED
            session.add(db_chunk)
            session.commit()
            logger.info(f"Chunk {chunk_id} indexed successfully")

    except Exception as e:
        with Session(engine, expire_on_commit=False) as session:
            db_chunk = session.get(DBChunk, chunk_id)
            if self.request.retries >= self.max_retries:
                # Retries exhausted
                error_msg = traceback.format_exc()
                logger.error(f"Max retries exceeded while indexing chunk {chunk_id}: {error_msg}")
                if db_chunk:
                    db_chunk.index_status = KgIndexStatus.FAILED
                    db_chunk.index_result = error_msg
                    session.add(db_chunk)
                    session.commit()
                raise MaxRetriesExceededError(f"Max retries exceeded: {e}")
            else:
                # Retry the task
                logger.error(f"Error while indexing chunk {chunk_id}: {e}, retrying...")
                session.rollback()  # Rollback any changes
                if db_chunk and db_chunk.index_status != KgIndexStatus.RUNNING:
                    db_chunk.index_status = KgIndexStatus.PENDING
                    session.add(db_chunk)
                    session.commit()
                raise self.retry(exc=e)
