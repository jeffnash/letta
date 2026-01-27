"""add memory_mode to agents

Revision ID: a7b8c9d0e1f2
Revises: 82feb220a9b8
Create Date: 2026-01-27 07:10:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "82feb220a9b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add memory_mode column to agents table
    # NULL means not yet evaluated (will trigger auto-migration on supported models)
    # 'system_prompt' = legacy mode (memory in system message)
    # 'context_message' = new mode (memory as separate message for better caching)
    op.add_column("agents", sa.Column("memory_mode", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("agents", "memory_mode")
