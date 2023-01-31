FROM tiangolo/uvicorn-gunicorn:python3.10

RUN apt update && \
    apt install -y htop libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt