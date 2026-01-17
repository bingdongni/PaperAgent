FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    postgresql-client \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/papers data/experiments data/literature data/outputs logs

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8000 8501

CMD ["python", "-m", "paperagent.cli"]
