# Use the official Playwright image (Pre-installed browsers & dependencies)
FROM mcr.microsoft.com/playwright/python:v1.49.1-jammy

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Render assigns (default 10000)
ENV PORT=10000
EXPOSE $PORT

# Run the app with Gunicorn
CMD gunicorn fact_check_api:app --bind 0.0.0.0:$PORT