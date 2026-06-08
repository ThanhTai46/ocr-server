FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr_server.py .

EXPOSE 8080
CMD ["gunicorn", "--preload", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "--max-requests", "100", "--max-requests-jitter", "20", "ocr_server:app"]
