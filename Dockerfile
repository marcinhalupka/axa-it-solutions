# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# To view container logs in realtime
ENV PYTHONUNBUFFERED 1

# Run the main.py when the container launches
CMD ["python", "main.py"]
