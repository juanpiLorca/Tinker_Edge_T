FROM python:3.8-slim

WORKDIR /workspace 

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    build-essential \
    libpython3-dev \
    python3-sklearn python3-sklearn-lib python3-sklearn-lib python-sklearn-doc
    
ENV ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/skleanr/__check_build/../../sckit_learn.libs/libgomp-d22c30c5.so.1.0.0

COPY . /workspace/

CMD ["python3", "test.py"]