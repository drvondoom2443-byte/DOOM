FROM python:3.10-slim

WORKDIR /app

# Copy requirements first so it doesn't use a broken cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of your code
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
