"""Fix: Add last_synced column to providers if missing

Revision ID: f1a2b3c4d5e6
Revises: a1b2c3d4e5f8
Create Date: 2026-02-05 09:18:40.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "a1b2c3d4e5f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if column exists before adding
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('providers')]
    
    if 'last_synced' not in columns:
        op.add_column("providers", sa.Column("last_synced", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Check if column exists before dropping
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('providers')]
    
    if 'last_synced' in columns:
        op.drop_column("providers", "last_synced")
