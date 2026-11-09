FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_THREAD_LIMIT=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-por \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10002 svc && useradd --uid 10002 --gid svc --create-home svc
WORKDIR /srv
COPY trainer/requirements.txt trainer/requirements.txt
RUN pip install --no-cache-dir -r trainer/requirements.txt
COPY service/__init__.py service/__init__.py
COPY service/app/__init__.py service/app/__init__.py
COPY service/app/vision service/app/vision
COPY trainer ./trainer
USER svc
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD python -m trainer.app.health
CMD ["python","-m","trainer.app.main"]
