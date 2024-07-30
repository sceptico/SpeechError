# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific directories and files into the container
COPY data/metadata/dataset.csv /app/data/metadata/dataset.csv
COPY experiments /app/experiments
COPY scripts /app/scripts
COPY src /app/src

# Start an interactive terminal
CMD ["bash"]
