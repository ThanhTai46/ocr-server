FROM python:3.13-slim

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr_server.py .

EXPOSE 8080
CMD ["python3", "ocr_server.py"]
