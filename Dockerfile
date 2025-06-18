FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

RUN apt update && apt upgrade -y && \
    apt install -y build-essential python3-venv && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt