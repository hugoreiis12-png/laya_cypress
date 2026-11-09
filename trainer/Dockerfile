FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10002 svc && useradd --uid 10002 --gid svc --create-home svc
WORKDIR /srv
COPY trainer/requirements.txt trainer/requirements.txt
RUN pip install --no-cache-dir -r trainer/requirements.txt
COPY trainer ./trainer
USER svc
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
  CMD python -m trainer.app.health
CMD ["python","-m","trainer.app.main"]
