FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Start app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
