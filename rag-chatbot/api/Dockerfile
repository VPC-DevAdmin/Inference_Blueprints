FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the app directory is on PYTHONPATH
ENV PYTHONPATH=/app

# Expose the port the service runs on
EXPOSE 5001

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001", "--reload"]
