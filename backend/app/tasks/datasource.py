from sqlmodel import Session, select, delete
from celery.utils.log import get_task_logger

from app.celery import app as celery_app
from app.core.db import engine
from app.models import (
    DataSource,
    Document,
    Chunk,
    Relationship,
    Entity,
)
import logging
from app.rag.datasource import get_data_source_loader
from app.repositories import data_source_repo
from .rag_build import build_vector_index_from_document


logger = get_task_logger(__name__)
# Set log level to DEBUG
logger.setLevel(logging.DEBUG)


@celery_app.task
def import_documents_from_datasource(data_source_id: int):
    print("\nNew process: Trying to import_documents_from_datasource")
    with Session(engine) as session:
        data_source = data_source_repo.get(session, data_source_id)
        if data_source is None:
            logger.error(f"Data source with id {data_source_id} not found")
            return

        loader = get_data_source_loader(
            session,
            data_source.data_source_type,
            data_source.id,
            data_source.user_id,
            data_source.config,
        )
        if not loader:
            logger.error(f"Failed to get data source loader for data_source_id {data_source_id}")
            return

        all_documents=list(loader.load_documents())
        for document in all_documents:
            try:
                logger.info(f"Processing document with id {document.id}")
                session.add(document)
                session.commit()
                logger.info(f"Document with id {document.id} added successfully.")
                build_vector_index_from_document.delay(data_source_id, document.id)
            except Exception as e:
                logger.error(f"Error committing document {document.id}: {e}")
                session.rollback()
            


@celery_app.task
def purge_datasource_related_resources(data_source_id: int):
    # delete all the related resources
    #   - documents
    #   - chunks
    #   - vector index
    #   - kg index
    with Session(engine) as session:
        data_source = session.get(DataSource, data_source_id)
        if data_source is None:
            logger.error(f"Data source with id {data_source_id} not found")
            return

        assert data_source.deleted_at is not None

        document_ids = session.exec(
            select(Document.id).where(Document.data_source_id == data_source_id)
        ).all()

        stmt = delete(Relationship).where(Relationship.document_id.in_(document_ids))
        session.exec(stmt)

        stmt = delete(Chunk).where(Chunk.document_id.in_(document_ids))
        session.exec(stmt)

        stmt = delete(Document).where(Document.data_source_id == data_source_id)
        session.exec(stmt)

        session.commit()
