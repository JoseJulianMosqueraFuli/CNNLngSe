FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Instalar dependencias del sistema necesarias para compilar paquetes de Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry.
RUN pip install --no-cache-dir poetry

# Copiar solo los archivos de dependencias primero para aprovechar la caché.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-cache

# Copiar el resto del código e instalar el paquete.
COPY . .
RUN poetry install --no-cache

ENTRYPOINT ["sign-classifier"]
CMD ["--help"]
