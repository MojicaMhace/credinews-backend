FROM mcr.microsoft.com/playwright/python:v1.49.1-jammy
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium
RUN playwright install-deps
RUN python -m nltk.downloader punkt stopwords
COPY . .
ENV PORT=10000
EXPOSE $PORT
CMD gunicorn fact_check_api:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --preload