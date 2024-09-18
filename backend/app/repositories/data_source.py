from typing import Optional
from datetime import datetime, UTC
import logging
from sqlmodel import select, Session, func
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import DataSource, Document, Chunk, Relationship
from app.repositories.base_repo import BaseRepo



# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level

class DataSourceRepo(BaseRepo):
    model_cls = DataSource

    def paginate(
        self,
        session: Session,
        params: Params | None = Params(),
    ) -> Page[DataSource]:
        # logger.debug(f"Paginating data sources with params: {params}")
        query = (
            select(DataSource)
            .where(DataSource.deleted_at == None)
            .order_by(DataSource.created_at.desc())
        )
        return paginate(session, query, params)

    def get(
        self,
        session: Session,
        data_source_id: int,
    ) -> Optional[DataSource]:
        # logger.debug(f"Fetching data source with id: {data_source_id}")
        return session.scalars(
            select(DataSource).where(
                DataSource.id == data_source_id, DataSource.deleted_at == None
            )
        ).first()

    def delete(self, session: Session, data_source: DataSource) -> None:
        data_source.deleted_at = datetime.now(UTC)
        session.add(data_source)
        session.commit()

    def overview(self, session: Session, data_source: DataSource) -> dict:
        data_source_id = data_source.id
        logger.debug(f"Generating overview for data source with id: {data_source_id}")
        documents_count = session.scalar(
            select(func.count(Document.id)).where(
                Document.data_source_id == data_source_id
            )
        )
        chunks_count = session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.document.has(Document.data_source_id == data_source_id)
            )
        )
        logger.debug(f"Chunks count for data source {data_source_id}: {chunks_count}")

        # Vector index status
        statement = (
            select(Document.index_status, func.count(Document.id))
            .where(Document.data_source_id == data_source_id)
            .group_by(Document.index_status)
            .order_by(Document.index_status)
        )
        results = session.execute(statement).all()
        vector_index_status = [{"index_status": row[0], "count": row[1]} for row in results]  # Convert to dict
        # vector_index_status = {s: c for s, c in results}
        # logger.debug(f"Vector index status for data source {data_source_id}: {vector_index_status}")

        overview_data = {
            "documents": {
                "total": documents_count,
            },
            "chunks": {
                "total": chunks_count,
            },
            "vector_index": vector_index_status,
        }

        if data_source.build_kg_index:
            # logger.debug(f"KG indexing enabled for data source {data_source_id}")
            # Relationship count for KG index
            relationships_count = session.scalar(
                select(func.count(Relationship.id)).where(
                    Relationship.document_id.in_(
                        select(Document.id).where(
                            Document.data_source_id == data_source_id
                        )
                    )
                )
            )
            # logger.debug(f"Relationship count for data source {data_source_id}: {relationships_count}")
            overview_data["relationships"] = {
                "total": relationships_count,
            }
            # KG index status
            statement = (
                select(Chunk.index_status, func.count(Chunk.id))
                .where(Chunk.document.has(Document.data_source_id == data_source_id))
                .group_by(Chunk.index_status)
                .order_by(Chunk.index_status)
            )
            results = session.execute(statement).all()
            # kg_index_status = {s: c for s, c in results}
            kg_index_status = [{"index_status": row[0], "count": row[1]} for row in results]  # Convert to dict
            logger.debug(f"KG index status for data source {data_source_id}: {kg_index_status}")
            overview_data["kg_index"] = kg_index_status
        logger.info(f"Overview generated for data source {data_source_id}")
        return overview_data


data_source_repo = DataSourceRepo()
