"""Create users, tokens and threads tables."""

from alembic import op
import sqlalchemy as sa

revision = "6c045eedbeb3"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("username", sa.Text(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )

    op.create_table(
        "tokens",
        sa.Column("token", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )

    op.create_table(
        "threads",
        sa.Column("thread_id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("threads")
    op.drop_table("tokens")
    op.drop_table("users")
