#!/bin/bash
set -e

# Install deps to run the API in a seperate venv to isolate different components
conda create --name api-env -y python=3.10 pip
$HOME/.conda/envs/api-env/bin/pip install fastapi==0.109.2 uvicorn[standard]==0.27.0.post1 python-multipart==0.0.7 langchain==0.0.335 langchain-community==0.0.19 openai==1.55.3 httpx==0.27.2 unstructured[all-docs]==0.17.2 sentence-transformers==2.7.0 llama-index==0.9.44 dataclass-wizard==0.22.3 pymilvus==2.3.1 opencv-python==4.8.0.76 hf_transfer==0.1.5 text_generation==0.6.1 transformers==4.40.0 nltk==3.8.1 torch==2.1.1 

# Install deps to run the UI in a seperate venv to isolate different components
conda create --name ui-env -y python=3.10 pip
$HOME/.conda/envs/ui-env/bin/pip install dataclass_wizard==0.22.2 gradio==4.15.0 jinja2==3.1.2 numpy==1.25.2 protobuf==3.20.3 PyYAML==6.0 uvicorn==0.22.0 torch==2.1.1 tiktoken==0.7.0 regex==2024.5.15 fastapi==0.112.2

sudo -E /opt/conda/bin/pip install anyio==4.3.0 pymilvus==2.3.1 transformers==4.40.0

sudo -E mkdir -p /mnt/milvus
sudo -E mkdir -p /data
sudo -E chown $NVWB_UID:$NVWB_GID /mnt/milvus
sudo -E chown $NVWB_UID:$NVWB_GID /data

sudo -E curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo -E bash
sudo -E apt-get install git-lfs
