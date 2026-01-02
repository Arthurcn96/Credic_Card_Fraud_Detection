# Use a base image with Python
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# We install dependencies here to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn specifically for the API
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy the entire project
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application
# We use -m src.app.main to ensure Python correctly handles module imports
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
