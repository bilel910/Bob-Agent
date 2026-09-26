# Use an appropriate base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Install system dependencies for building libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency management files (lock file and pyproject.toml) first
COPY uv.lock pyproject.toml README.md /app/

# The large CUDA wheels torch pulls in (e.g. nvidia-nccl-cu12, several hundred MB)
# regularly exceed uv's 30s default HTTP timeout and fail the build. Give them room.
ENV UV_HTTP_TIMEOUT=300

# Install the application dependencies
RUN uv sync --frozen --no-cache

# Copy your application code into the container
COPY src/ /app/

# Set the virtual environment environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .

# Define volumes
VOLUME ["/app/data"]

# Expose the port
EXPOSE 8080

# Run the FastAPI app using uvicorn
CMD ["/app/.venv/bin/fastapi", "run", "ai_companion/interfaces/whatsapp/webhook_endpoint.py", "--port", "8080", "--host", "0.0.0.0"]
