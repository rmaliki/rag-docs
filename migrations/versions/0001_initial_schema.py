"""Initial schema: documents, ACLs, chunks with pgvector + full-text search, row-level security.

Revision ID: 0001
Revises:
Create Date: 2026-10-21
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# The API connects as this role. It is created outside migrations
# (docker/postgres/init or by your DBA) because creating roles needs elevated rights.
APP_ROLE = "rag_app"

UPGRADE = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE documents (
        id              varchar(128) PRIMARY KEY,
        title           text         NOT NULL,
        department      varchar(64)  NOT NULL,
        classification  varchar(32)  NOT NULL,
        version         integer      NOT NULL CHECK (version >= 1),
        is_public       boolean      NOT NULL DEFAULT false,
        source_uri      text         NOT NULL,
        content_hash    varchar(64)  NOT NULL,
        created_at      timestamptz  NOT NULL DEFAULT now(),
        updated_at      timestamptz  NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE document_acl (
        document_id  varchar(128) NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        principal    varchar(256) NOT NULL,
        PRIMARY KEY (document_id, principal)
    )
    """,
    "CREATE INDEX ix_document_acl_principal ON document_acl (principal)",
    """
    CREATE TABLE document_chunks (
        id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        document_id    varchar(128) NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        chunk_index    integer      NOT NULL,
        page           integer,
        content        text         NOT NULL,
        embedding      vector(384)  NOT NULL,
        search_vector  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
        UNIQUE (document_id, chunk_index)
    )
    """,
    "CREATE INDEX ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX ix_document_chunks_search_vector ON document_chunks USING gin (search_vector)",
    # The principals of the current request, set per transaction by the API:
    #   SELECT set_config('app.principals', '["group:hr-team","user:<oid>"]', true)
    # Unset or empty means no principals, so restricted rows are invisible by default.
    """
    CREATE FUNCTION app_principals() RETURNS text[]
    LANGUAGE sql STABLE AS $$
        SELECT ARRAY(
            SELECT jsonb_array_elements_text(
                NULLIF(current_setting('app.principals', true), '')::jsonb
            )
        )
    $$
    """,
    # Row-level security is the database-level safety net under the application's own
    # ACL filter: even a buggy query run as rag_app cannot read unauthorized rows.
    # The table owner (rag_admin, used for ingestion) is not subject to these policies.
    "ALTER TABLE documents ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE document_acl ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY",
    """
    CREATE POLICY document_acl_read ON document_acl FOR SELECT
    USING (principal = ANY (app_principals()))
    """,
    """
    CREATE POLICY documents_read ON documents FOR SELECT
    USING (
        is_public
        OR EXISTS (
            SELECT 1 FROM document_acl a
            WHERE a.document_id = documents.id AND a.principal = ANY (app_principals())
        )
    )
    """,
    """
    CREATE POLICY document_chunks_read ON document_chunks FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = document_chunks.document_id
              AND (
                  d.is_public
                  OR EXISTS (
                      SELECT 1 FROM document_acl a
                      WHERE a.document_id = d.id AND a.principal = ANY (app_principals())
                  )
              )
        )
    )
    """,
    f"""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
            GRANT USAGE ON SCHEMA public TO {APP_ROLE};
            GRANT SELECT ON documents, document_acl, document_chunks TO {APP_ROLE};
            GRANT EXECUTE ON FUNCTION app_principals() TO {APP_ROLE};
        ELSE
            RAISE WARNING 'role {APP_ROLE} does not exist: grant it read access before starting the API';
        END IF;
    END
    $$
    """,
]

DOWNGRADE = [
    "DROP TABLE IF EXISTS document_chunks",
    "DROP TABLE IF EXISTS document_acl",
    "DROP TABLE IF EXISTS documents",
    "DROP FUNCTION IF EXISTS app_principals()",
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
