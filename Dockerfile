# LiftLab image, one image, two compose roles (sim/pipeline + Streamlit UI).
FROM python:3.11-slim-bookworm

# uv, pinned for reproducibility (matches the local toolchain).
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- Dependency layer (cached unless lock/pyproject change) -----------------
# Install deps without the project first so editing source doesn't bust the
# dependency cache. The `ui` group brings in Streamlit for the report card.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --group ui --group causal

# --- Project layer ----------------------------------------------------------
COPY . .
RUN uv sync --frozen --group ui --group causal

# Put the venv on PATH so `streamlit`, `dbt`, `liftlab` resolve directly.
ENV PATH="/app/.venv/bin:$PATH" \
    LIFTLAB_DUCKDB=/app/data/warehouse/liftlab.duckdb \
    DBT_PROFILES_DIR=/app/dbt

EXPOSE 8501

# Default role: serve the Streamlit report card. The compose `pipeline` service
# overrides this command to build data + run the simulation/eval first.
CMD ["streamlit", "run", "src/liftlab/ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
