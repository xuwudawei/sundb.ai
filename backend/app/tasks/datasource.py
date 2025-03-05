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
    Image,
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
            # Step 1: Delete all documents associated with the data source
            document_ids = session.exec(
                select(Document.id).where(Document.data_source_id == data_source_id)
            ).all()

            if document_ids:
                # Step 2: Delete all relationships tied to these documents
                stmt = delete(Relationship).where(Relationship.document_id.in_(document_ids))
                session.exec(stmt)
                session.commit()  # Commit after deleting relationships
                print(f"Deleted relationships for documents tied to data source {data_source_id}.")

                # Step 3: Delete all chunks tied to these documents
                stmt = delete(Chunk).where(Chunk.document_id.in_(document_ids))
                session.exec(stmt)
                session.commit()  # Commit after deleting chunks
                print(f"Deleted chunks for documents tied to data source {data_source_id}.")

                # Step 4: Delete all entities linked to these documents through relationships
                related_entities_ids = session.exec(
                    select(Entity.id).where(
                        Entity.id.in_(select(Relationship.source_entity_id).where(Relationship.document_id.in_(document_ids))) |
                        Entity.id.in_(select(Relationship.target_entity_id).where(Relationship.document_id.in_(document_ids)))
                    )
                ).all()

                if related_entities_ids:
                    stmt = delete(Entity).where(Entity.id.in_(related_entities_ids))
                    session.exec(stmt)
                    session.commit()  # Commit after deleting related entities
                    print(f"Deleted related entities for documents tied to data source {data_source_id}.")

            # Step 5: Delete all orphaned relationships that are no longer tied to any document
            orphaned_relationships = session.exec(
                select(Relationship.id).where(
                    ~Relationship.document_id.in_(select(Document.id))
                )
            ).all()

            if orphaned_relationships:
                stmt = delete(Relationship).where(Relationship.id.in_(orphaned_relationships))
                session.exec(stmt)
                session.commit()  # Commit after deleting orphaned relationships
                print(f"Deleted orphaned relationships for data source {data_source_id}.")

            # Step 6: Delete all orphaned entities that are no longer referenced by any relationships
            orphaned_entity_ids = session.exec(
                select(Entity.id).where(
                    ~Entity.id.in_(select(Relationship.source_entity_id)) &
                    ~Entity.id.in_(select(Relationship.target_entity_id))
                )
            ).all()

            if orphaned_entity_ids:
                stmt = delete(Entity).where(Entity.id.in_(orphaned_entity_ids))
                session.exec(stmt)
                session.commit()  # Commit after deleting orphaned entities
                print(f"Deleted orphaned entities for data source {data_source_id}.")

            
            # Step 7: Delete all images tied to these documents.
            stmt = delete(Image).where(Image.source_document_id.in_(document_ids))
            session.exec(stmt)
            session.commit()  # Commit after deleting images.
            print(f"Deleted images for documents tied to data source {data_source_id}.")

            # Step 8: Delete all documents tied to the data source
            stmt = delete(Document).where(Document.data_source_id == data_source_id)
            session.exec(stmt)
            session.commit()  # Commit after deleting documents
            print(f"Deleted documents for data source {data_source_id}.")

            print(f"Purged all resources for data source {data_source_id}.")