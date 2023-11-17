# Use the official Python image as the base image
FROM python:3.8.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the Python files into the container
COPY . .

# Set the default command to run when the container starts
CMD ["python3", "main.py"]