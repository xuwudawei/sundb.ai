from sqlmodel import Session, select, delete
from app.core.db import engine
from app.models import Entity, Relationship, Document, Chunk

def purge_all_related_resources(data_source_id: int):
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

        # Step 7: Delete all documents tied to the data source
        stmt = delete(Document).where(Document.data_source_id == data_source_id)
        session.exec(stmt)
        session.commit()  # Commit after deleting documents
        print(f"Deleted documents for data source {data_source_id}.")

        print(f"Purged all resources for data source {data_source_id}.")

# Call this function to clean up everything tied to a data source
purge_all_related_resources(data_source_id=1)
