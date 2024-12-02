import sqlalchemy
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from enum import Enum

class DistanceMetric(Enum):
    L2 = 'vector_l2_ops'
    COSINE = 'vector_cosine_ops'
    INNER_PRODUCT = 'vector_ip_ops'

    def to_operator_class(self):
        return self.value

class PgVectorAdaptor:
    """
    A wrapper over existing SQLAlchemy engine to provide additional vector search capabilities.
    """

    engine: sqlalchemy.engine.Engine

    def __init__(self, engine: sqlalchemy.engine.Engine):
        self.engine = engine

    def _check_vector_column(self, column: Column):
        if not isinstance(column.type, Vector):
            raise ValueError("Not a vector column")

    def has_vector_index(self, column: Column) -> bool:
        """
        Check if the index for the vector column exists.
        """

        self._check_vector_column(column)

        with self.engine.connect() as conn:
            table_name = column.table.name
            index_name = f"vec_idx_{column.name}"
            query = sqlalchemy.text("""
                SELECT 1
                FROM pg_indexes
                WHERE tablename = :table_name AND indexname = :index_name
            """)
            result = conn.execute(query, {"table_name": table_name, "index_name": index_name}).fetchone()
            return result is not None

    def create_vector_index(
        self,
        column: Column,
        distance_metric: DistanceMetric,
        skip_existing: bool = False,
    ):
        """
        Create vector index for the vector column.

        Parameters
        ----------
        column : sqlalchemy.Column
            The column for which the vector index is to be created.

        distance_metric : DistanceMetric
            The distance metric to be used for the vector index.

        skip_existing : bool
            If True, skips creating the index if it already exists. Default is False.

        Raises
        ------
        ValueError
            If the vector column does not have a fixed dimension.

        ValueError
            If the column is not a vector column.
        """

        self._check_vector_column(column)

        if column.type.dim is None:
            raise ValueError(
                "Vector index is only supported for fixed dimension vectors"
            )

        if skip_existing and self.has_vector_index(column):
            # TODO: Verify whether the distance metric is correct if necessary
            return

        with self.engine.connect() as conn:
            table_name = column.table.name
            column_name = column.name
            index_name = f"vec_idx_{column_name}"

            operator_class = distance_metric.to_operator_class()

            sql = f"""
            CREATE INDEX {index_name}
            ON {table_name}
            USING hnsw ({column_name} {operator_class});
            """

            conn.execute(sqlalchemy.text(sql))
