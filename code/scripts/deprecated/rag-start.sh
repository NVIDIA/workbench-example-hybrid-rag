#!/bin/bash

# This script spins up the API server and the Milvus vector store

# Start milvus
echo "Starting Milvus"
/opt/conda/bin/milvus-server --data /mnt/milvus/ &
pid1=$!

# Start API
echo "Starting API"
cd /project/code/ && /opt/conda/envs/api-env/bin/python -m uvicorn chain_server.server:app --port=8000 --host='0.0.0.0' &
pid2=$!

# Now, wait for each command to complete
if ! wait $pid1; then
    echo "Milvus failed"
    exit 1
fi

if ! wait $pid2; then
    echo "API failed"
    exit 1
fi

sleep 3

echo "RAG system ready"
