# Use the official Python image as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install FastAPI and any other dependencies
RUN pip install -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define the command to run your FastAPI app
CMD ["uvicorn", "main:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
