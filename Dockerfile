
# Instructions for building the Doc container

version: '3.8'

services:
  alzheimer-api:
    build: .
    container_name: alzheimer-api-container
    ports:
      - "5000:5000"
    volumes:
      - ./src:/app/src
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/alzheimer_cnn.h5
      - DATA_PATH=/app/data/processed/audio_features.csv
    depends_on:
      - db

  db:
    image: postgres:13
    container_name: alzheimer-db
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: alzheimer_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:

# Dockerfile for API Service
  alzheimer-api:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      - db


# It includes:

# Base Image: Uses Python 3.9 as the base environment.
# Working Directory: Sets /app as the default working directory.
# Dependencies: Installs required Python packages from requirements.txt.
# Copies Source Code: Moves all necessary project files into the container.
# Exposes API Port: Maps container port 5000 for API access.
# Runs the API: Specifies the Flask app to start when the container runs.