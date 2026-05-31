FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build tools needed for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the embedding model during Docker build
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"


# Copy application files
COPY . .

# Expose the default port for Hugging Face Spaces (7860) or Render ($PORT)
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
