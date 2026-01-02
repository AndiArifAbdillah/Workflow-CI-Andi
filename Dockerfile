# Gunakan base image python
FROM python:3.9-slim

# Set folder kerja
WORKDIR /app

# Copy requirements dan install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy seluruh kode ke dalam image
COPY . .

# Perintah yang dijalankan saat container nyala
CMD ["python", "main.py"]