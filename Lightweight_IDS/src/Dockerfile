FROM python:3.7-slim

WORKDIR /workspace 

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpython3-dev \
    python3-sklearn \
    python3-sklearn-lib \
    python-sklearn-doc \
    && rm -rf /var/lib/apt/lists/*
    
# ENV LD_PRELOAD=/usr/local/lib/python3.7/dist-packages/skleanr/__check_build/../../sckit_learn.libs/libgomp-d22c30c5.so.1.0.0

RUN pip install --upgrade pip && \
    pip install scikit-learn

COPY . /workspace/

CMD ["python3", "test.py"]


