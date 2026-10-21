#!/bin/sh
# Runs once, on first database initialisation.
# rag_app is the API's least-privileged role: no superuser, no BYPASSRLS,
# and (after the migration's grants) SELECT only. Row-level security applies to it.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v app_password="$RAG_APP_PASSWORD" <<'EOSQL'
CREATE ROLE rag_app LOGIN PASSWORD :'app_password' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
