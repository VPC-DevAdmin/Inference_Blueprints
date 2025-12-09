FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default model, can be overridden when you run the container
ENV EMBEDDINGS_MODEL_NAME=BAAI/bge-base-en-v1.5

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
