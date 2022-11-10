FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.9

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y \
    build-essential curl htop jq tree less

WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN pip install --upgrade wheel && \
    pip install -r requirements.txt 

# to cache the pretraining models
COPY cache_huggingface_models.py /app/
RUN python cache_huggingface_models.py 

# Ensure easy remote debugging in VSCode
RUN pip install --upgrade debugpy && \
    pip install git+https://github.com/alexpolozov/debugpy-run.git@fd3c79532d772932081ce55a1c5f641a43fdbdb9
