"""Initial migration

Revision ID: eaa44b944665
Revises: 
Create Date: 2023-12-12 00:40:27.117992

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'eaa44b944665'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('your_model',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('RESPONSE', sa.String(length=255), nullable=True),
    sa.Column('LEVEL', sa.String(length=255), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('data_training')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('data_training',
    sa.Column('RESPONSE', mysql.TEXT(), nullable=False),
    sa.Column('LEVEL', mysql.TEXT(), nullable=False),
    mysql_collate='utf8mb4_general_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.drop_table('your_model')
    # ### end Alembic commands ###