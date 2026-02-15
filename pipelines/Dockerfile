FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU-only first (avoids downloading ~3GB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install the rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/pipelines

WORKDIR /app/pipelines

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9099"]
