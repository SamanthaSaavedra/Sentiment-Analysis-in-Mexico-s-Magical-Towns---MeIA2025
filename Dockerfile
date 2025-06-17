FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

RUN apt update && apt upgrade -y && \
    apt install -y build-essential python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt