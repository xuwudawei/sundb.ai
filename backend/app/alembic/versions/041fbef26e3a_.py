"""empty message

Revision ID: 041fbef26e3a
Revises: 04d81be446c3
Create Date: 2024-08-19 08:20:13.695891

"""

from alembic import op
from sqlalchemy.dialects.mysql import DATETIME


# revision identifiers, used by Alembic.
revision = "041fbef26e3a"
down_revision = "04d81be446c3"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "site_settings",
        "created_at",
        existing_type=DATETIME(timezone=True, fsp=6),
        nullable=False,
    )
    op.alter_column(
        "site_settings",
        "updated_at",
        existing_type=DATETIME(timezone=True, fsp=6),
        nullable=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "site_settings",
        "created_at",
        existing_type=DATETIME(timezone=True),
        nullable=False,
    )
    op.alter_column(
        "site_settings",
        "updated_at",
        existing_type=DATETIME(timezone=True),
        nullable=False,
    )
    # ### end Alembic commands ###
