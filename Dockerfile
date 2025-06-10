FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV and Flask/Gunicorn
RUN pip install opencv-python-headless

WORKDIR /app
COPY . /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# EXPOSE the port Flask will run on
EXPOSE 5000

# Production-grade CMD using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]

