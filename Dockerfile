FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the models into the image: no download at startup, works without internet egress.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
    && chmod -R a+rX /opt/hf-cache

RUN useradd --create-home --uid 10001 app
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY eval ./eval
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
