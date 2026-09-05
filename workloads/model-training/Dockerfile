FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -e .

COPY train.py ./

ENTRYPOINT ["python", "train.py"]