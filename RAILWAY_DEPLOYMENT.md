# Railway Deployment Guide for Letta Server with CLIProxy Support

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI** (optional): `npm install -g @railway/cli`
3. **GitHub Repository**: Push your modified letta-server code to GitHub

## Step 1: Set Up the Project on Railway

### Option A: Deploy via Railway Dashboard (Recommended)

1. Go to [railway.app/new](https://railway.app/new)
2. Click "Deploy from GitHub repo"
3. Select your forked/modified letta-server repository
4. Railway will auto-detect the Dockerfile

### Option B: Deploy via Railway CLI

```bash
railway login
railway init
railway up
```

## Step 2: Add PostgreSQL Database

Railway has a native PostgreSQL addon with pgvector support:

1. In your Railway project, click "+ New"
2. Select "Database" → "PostgreSQL"
3. Wait for the database to provision
4. Railway automatically sets `DATABASE_URL` environment variable

## Step 3: Configure Environment Variables

In Railway Dashboard → Variables, add:

### Required Variables

```bash
# Database (auto-set if you use Railway Postgres)
LETTA_PG_URI=${{Postgres.DATABASE_URL}}

# OpenAI API Key (for embeddings ONLY)
OPENAI_API_KEY=sk-proj-YOUR_REAL_OPENAI_KEY

# CLIProxyAPI Configuration (for LLM inference)
CLIPROXY_API_KEY=your_cliproxy_api_key_here
CLIPROXY_BASE_URL=https://your-cliproxy-instance.up.railway.app/v1
```

### Optional Variables

```bash
# Server Config
PORT=8283
HOST=0.0.0.0
LETTA_DEBUG=false

# Redis (Railway manages this internally with our Dockerfile)
# LETTA_REDIS_HOST=localhost
```

## Step 4: Configure Build Settings

In Railway Dashboard → Settings:

1. **Build Command**: (leave empty, uses Dockerfile)
2. **Dockerfile Path**: `Dockerfile.railway`
3. **Watch Paths**: Leave default

Or use the `railway.toml` we created.

## Step 5: Configure Networking

1. Go to Settings → Networking
2. Click "Generate Domain" to get a public URL
3. Note this URL - you'll use it to connect Letta Code

Your server will be available at something like:
```
https://letta-server-production-xxxx.up.railway.app
```

## Step 6: After Deployment - Add CLIProxy as a Provider

Once your server is running, you need to register CLIProxy as a BYOK provider via the API.

### Using curl:

```bash
# Replace YOUR_LETTA_SERVER_URL with your Railway URL
LETTA_URL="https://letta-server-production-xxxx.up.railway.app"

# Create CLIProxy provider
curl -X POST "${LETTA_URL}/v1/providers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cliproxy",
    "provider_type": "openai",
    "api_key": "your_cliproxy_api_key_here",
    "base_url": "https://your-cliproxy-instance.up.railway.app/v1"
  }'
```

### Using Python:

```python
from letta_client import Letta

client = Letta(base_url="https://letta-server-production-xxxx.up.railway.app")

# Create CLIProxy provider
provider = client.providers.create(
    name="cliproxy",
    provider_type="openai", 
    api_key="your_cliproxy_api_key_here",
    base_url="https://your-cliproxy-instance.up.railway.app/v1"
)
print(f"Created provider: {provider.id}")
```

## Step 7: Configure Letta Code

Update your Letta Code settings to point to your Railway server.

In your Letta Code directory, create/update `.letta/config.json`:

```json
{
  "server_url": "https://letta-server-production-xxxx.up.railway.app",
  "default_llm": "cliproxy/gpt-5.2-medium",
  "default_embedding": "openai/text-embedding-3-small"
}
```

Or set environment variables:

```bash
export LETTA_SERVER_URL="https://letta-server-production-xxxx.up.railway.app"
```

## Architecture Overview

```
┌─────────────────┐
│   Letta Code    │  (CLI on your machine)
│     (Client)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│        Railway-hosted Letta Server              │
│  ┌──────────────────────────────────────────┐   │
│  │  API Server (port 8283)                  │   │
│  │    - Agent management                    │   │
│  │    - Memory & conversation              │   │
│  │    - Tool execution                      │   │
│  └──────────────┬────────────────┬──────────┘   │
│                 │                │               │
│                 ▼                ▼               │
│  ┌──────────────────┐  ┌─────────────────────┐  │
│  │    PostgreSQL    │  │       Redis         │  │
│  │   (pgvector)     │  │    (sessions)       │  │
│  └──────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────┘
         │                         │
         │ Embeddings              │ LLM Inference
         ▼                         ▼
┌─────────────────┐       ┌─────────────────────┐
│   OpenAI API    │       │    CLIProxyAPI      │
│  (embeddings)   │       │ (GPT-5.2, Gemini,   │
│                 │       │  Claude, Qwen)      │
└─────────────────┘       └─────────────────────┘
```

## Troubleshooting

### Check Logs

```bash
railway logs
```

### Health Check

```bash
curl https://your-letta-server.up.railway.app/v1/health
```

### List Available Models

```bash
curl https://your-letta-server.up.railway.app/v1/models
```

### Database Connection Issues

Make sure `LETTA_PG_URI` is correctly set:
```bash
# In Railway, reference the Postgres addon like this:
LETTA_PG_URI=${{Postgres.DATABASE_URL}}
```

### Memory Issues

If you get OOM errors, upgrade your Railway plan or adjust:
- Reduce worker count
- Use smaller model context windows

## Cost Estimate

Railway pricing (as of 2024):
- **Hobby Plan**: $5/month + usage
- **Pro Plan**: $20/month + usage

Typical Letta server usage:
- ~512MB-1GB RAM
- Minimal CPU when idle
- Database storage grows with conversations

Estimated cost: **$10-30/month** for light to moderate use
