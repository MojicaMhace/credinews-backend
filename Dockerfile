# Use the official Playwright Python image (includes Python, Chromium, and dependencies)
FROM mcr.microsoft.com/playwright/python:v1.49.1-jammy

# Set the working directory
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install your Python libraries (Flask, Gunicorn, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Render assigns
ENV PORT=10000
EXPOSE $PORT

# Start the application
CMD gunicorn fact_check_api:app --bind 0.0.0.0:$PORT