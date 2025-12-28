#!/bin/bash
set -e

# NOTE: On Railway, the container filesystem is ephemeral across redeploys.
# To persist agents/data while running the bundled Postgres, you must attach
# a Railway Volume mounted at $PGDATA (default: /var/lib/postgresql/data).
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PGLOG="${PGLOG:-/var/lib/postgresql/postgresql.log}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8283}"

echo "=== Railway Letta Startup ==="
echo "PORT=$PORT, HOST=$HOST"
echo "PGDATA=$PGDATA"

# Start PostgreSQL
echo "Starting internal PostgreSQL..."

# Ensure PGDATA exists (Railway Volume should be mounted here)
mkdir -p "$PGDATA"
chown -R postgres:postgres "$PGDATA" || true

if [ ! -d "$PGDATA/base" ]; then
    echo "Initializing PostgreSQL database..."

    # initdb requires ownership of the data directory
    chown -R postgres:postgres "$PGDATA"
    su postgres -c "/usr/lib/postgresql/15/bin/initdb -D $PGDATA"

    su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA -l $PGLOG start"
    sleep 3

    su postgres -c "psql -c \"CREATE USER letta WITH PASSWORD 'letta' SUPERUSER;\"" || true
    su postgres -c "psql -c \"CREATE DATABASE letta OWNER letta;\"" || true
    su postgres -c "psql -d letta -c \"CREATE EXTENSION IF NOT EXISTS vector;\"" || true

    echo "PostgreSQL initialized with pgvector extension"
else
    echo "Starting existing PostgreSQL..."
    su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D $PGDATA -l $PGLOG start"
    sleep 2
fi

# Wait for postgres to be ready
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if pg_isready -U letta -h localhost -q; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "Attempt $i/30..."
    sleep 1
done

# Start Redis
echo "Starting Redis..."
redis-server --daemonize yes --bind 0.0.0.0

# Wait for Redis
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if redis-cli ping 2>/dev/null | grep -q PONG; then
        echo "Redis is ready!"
        break
    fi
    echo "Attempt $i/30..."
    sleep 1
done

# Set environment variables
# If you want to use Railway managed Postgres instead, set LETTA_PG_URI in Railway Variables.
# For the bundled internal Postgres (in this container), the default is localhost.
export LETTA_PG_URI="${LETTA_PG_URI:-postgresql://letta:letta@localhost:5432/letta}"
export LETTA_REDIS_HOST="${LETTA_REDIS_HOST:-localhost}"

echo "LETTA_PG_URI=$LETTA_PG_URI"
echo "LETTA_REDIS_HOST=$LETTA_REDIS_HOST"

# Run database migrations
echo "Running database migrations..."
cd /app
if ! alembic upgrade head; then
    echo "ERROR: Database migration failed!"
    exit 1
fi
echo "Database migration completed successfully."

# Debug: check if letta command exists
echo "Checking letta command..."
which letta || echo "letta not found in PATH"
echo "PATH=$PATH"
ls -la /app/.venv/bin/ | head -20

# Start Letta server
echo "=== Starting Letta Server at http://$HOST:$PORT ==="
echo "Executing: letta server --host $HOST --port $PORT"

# Use unbuffered output
export PYTHONUNBUFFERED=1
exec letta server --host "$HOST" --port "$PORT" 2>&1
