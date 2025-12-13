# Use a lightweight, standard Python image instead of the huge Playwright one
FROM python:3.10-slim

WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (no need for playwright install)
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copy the rest of the application
COPY . .

# Set port
ENV PORT=10000
EXPOSE $PORT

# Run the application with Gunicorn
CMD gunicorn fact_check_api:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 4 --worker-class gthread