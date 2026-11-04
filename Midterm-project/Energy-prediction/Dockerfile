
# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model
COPY predict.py .
COPY model_rfr .

# Expose application port
EXPOSE 9696

# Start Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "1", "predict:app"]
