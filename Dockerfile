FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_THREAD_LIMIT=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-por libzbar0 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10002 svc && useradd --uid 10002 --gid svc --create-home svc
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY service ./service
USER svc
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8020/healthz',timeout=4).status==200 else 1)"
CMD ["uvicorn","service.app.main:create_app","--factory","--host","0.0.0.0","--port","8020"]