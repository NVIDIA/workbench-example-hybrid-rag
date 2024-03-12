#!/bin/bash

# This script stops the API server and the Milvus vector store

pkill -SIGINT -f '$HOME/.local/bin/milvus-server'
pkill -SIGINT -f '^$HOME/.conda/envs/api-env/bin/python -m uvicorn chain_server.server:app'