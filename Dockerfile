# Use the official Python image from the Docker Hub
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Install a specific version of pip
RUN python -m pip install --upgrade pip==24.3.1

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip cache purge && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]