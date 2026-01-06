FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --upgrade pip && pip install uv

# Copy project metadata first for caching
COPY pyproject.toml uv.lock README.md /app/

# Copy code (needed so the project can be installed)
COPY src /app/src
COPY scripts /app/scripts

# Install deps + install the local project (because [tool.uv].package=true)
RUN uv sync --frozen

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "train"]


