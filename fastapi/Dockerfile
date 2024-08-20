# Use the official Python image from the Docker Hub
FROM python:3.10.14

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
    
    # install cmake
    WORKDIR /root
    RUN mkdir temp
    WORKDIR /root/temp
    RUN curl -OL https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4.tar.gz
    RUN tar -xzvf cmake-3.27.4.tar.gz
    
    WORKDIR /root/temp/cmake-3.27.4
    RUN ./bootstrap -- -DCMAKE_BUILD_TYPE:STRING=Release
    RUN make -j4
    RUN make install
    
    WORKDIR /root
    RUN rm -rf temp
    
    # mysql-connector install
    RUN pip install mysql-connector-python
    
    # Install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt



# Copy source code
COPY . .

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
