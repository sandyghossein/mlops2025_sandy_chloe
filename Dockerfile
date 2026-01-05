# Use Python 3.11 slim
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only files needed for installing dependencies first (better caching)
COPY pyproject.toml uv.lock /app/

# Install uv
RUN pip install --upgrade pip \
    && pip install uv

# Install project dependencies via uv
RUN uv sync --frozen

# Copy project files
COPY src /app/src
COPY scripts /app/scripts

# Install ml-project package
RUN uv python install .

# Set environment variables
ENV PYTHONPATH=/app/src

# Default command (can be overridden by docker-compose)
CMD ["uv", "run", "train"]
