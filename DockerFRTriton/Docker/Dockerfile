FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Workdir inside container
WORKDIR /app

# Copy code and model repository
COPY . /app

# Install Python deps for FastAPI wrapper
RUN pip install --no-cache-dir -r requirements.txt

# Default ports for Triton (8000/8001/8002) and FastAPI (3000)
EXPOSE 8000 8001 8002 3000

# Launch Triton and FastAPI together
CMD ["bash", "/app/Docker/start.sh"]
