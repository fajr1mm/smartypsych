# Use the official Python image as a parent image
FROM python:3.8.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install FastAPI and any other dependencies
RUN pip install -r requirements.txt

# Define the command to run your FastAPI app
CMD uvicorn fastapi_app:app --port=8000 --host0.0.0.0
