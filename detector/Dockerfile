# Build from the project root: docker build -f detector/Dockerfile -t laya-detector:local .
FROM python:3.12-slim
WORKDIR /app
COPY detector/requirements.txt detector/requirements.txt
RUN pip install --no-cache-dir -r detector/requirements.txt
COPY service/__init__.py service/__init__.py
COPY service/app/__init__.py service/app/__init__.py
COPY service/app/vision service/app/vision
COPY detector detector
ENV HF_HOME=/models
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Mo-Awadalla/legaldocuman-rfdetr', 'checkpoint_best_total.pth')"
EXPOSE 8030
CMD ["uvicorn", "detector.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8030", "--workers", "1"]
