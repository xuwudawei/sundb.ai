"""multiple_kb

Revision ID: d2ad44deab20
Revises: c7f016a904c1
Create Date: 2024-11-15 09:51:42.493749

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
# from tidb_vector.sqlalchemy import VectorType
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'd2ad44deab20'
down_revision = 'c7f016a904c1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('knowledge_bases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('index_methods', sa.JSON(), nullable=True),
        sa.Column('llm_id', sa.Integer(), nullable=True),
        sa.Column('embedding_model_id', sa.Integer(), nullable=True),
        sa.Column('documents_total', sa.Integer(), nullable=False),
        sa.Column('data_sources_total', sa.Integer(), nullable=False),
        sa.Column('created_by', sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_by', sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('deleted_by', sqlmodel.sql.sqltypes.GUID(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['deleted_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ),
        sa.ForeignKeyConstraint(['llm_id'], ['llms.id'], ),
        sa.ForeignKeyConstraint(['updated_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_table('knowledge_base_datasources',
        sa.Column('knowledge_base_id', sa.Integer(), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ),
        sa.ForeignKeyConstraint(['knowledge_base_id'], ['knowledge_bases.id'], ),
        sa.PrimaryKeyConstraint('knowledge_base_id', 'data_source_id')
    )
    op.add_column('documents', sa.Column('knowledge_base_id', sa.Integer(), nullable=True))
    op.create_foreign_key("fk_d_on_data_source_id", 'documents', 'data_sources', ['data_source_id'], ['id'])
    op.create_foreign_key("fk_d_on_knowledge_base_id", 'documents', 'knowledge_bases', ['knowledge_base_id'], ['id'])
    op.add_column('embedding_models', sa.Column('vector_dimension', sa.Integer(), nullable=False,server_default='0'))#modified by david
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('embedding_models', 'vector_dimension')
    op.drop_constraint("fk_d_on_data_source_id", 'documents', type_='foreignkey')
    op.drop_constraint("fk_d_on_knowledge_base_id", 'documents', type_='foreignkey')
    op.drop_column('documents', 'knowledge_base_id')
    op.drop_table('knowledge_base_datasources')
    op.drop_table('knowledge_bases')
    # ### end Alembic commands ###
