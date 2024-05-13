#!/bin/bash

# Check if gedit is running
# -x flag only match processes whose name (or command line if -f is
# specified) exactly match the pattern. 

if pgrep -x "milvus" > /dev/null
then
    # Check if the status is not 200
    if [[ $(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "http://localhost:8000/health") -ne 200 ]]; then
        echo "Error: 'http://localhost:8000/health' returned HTTP code $(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "http://localhost:8000/health")"
        exit 1
    fi

    # Check if the status is not 200
    if [[ $(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "http://localhost:19530/v1/vector/collections") -ne 200 ]]; then
        echo "Error: 'http://localhost:19530/v1/vector/collections' returned HTTP code $(curl -o /dev/null -s -w "%{http_code}" --max-time 3 'http://localhost:19530/v1/vector/collections')"
        exit 2
    fi
    
    echo "All URLs returned HTTP code 200"
    exit 0
else
    # Start milvus
    echo "Starting Milvus"
    $HOME/.local/bin/milvus-server --data /mnt/milvus/ &
    
    # Start API
    echo "Starting API"
    cd /project/code/ && $HOME/.conda/envs/api-env/bin/python -m uvicorn chain_server.server:app --port=8000 --host='0.0.0.0' &

    # Wait for service to be reachable.
    ATTEMPTS=0
    MAX_ATTEMPTS=20
    
    while [ $(curl -o /dev/null -s -w "%{http_code}" "http://localhost:8000/health") -ne 200 ]; 
    do 
      ATTEMPTS=$(($ATTEMPTS+1))
      if [ ${ATTEMPTS} -eq ${MAX_ATTEMPTS} ]
      then
        echo "Max attempts reached: $MAX_ATTEMPTS. Server may have timed out. Stop the container and try again. "
        exit 1
      fi
      
      echo "Polling inference server. Awaiting status 200; trying again in 5s. "
      sleep 5
    done 
    
    echo "Service reachable. Happy chatting!"
    exit 2
fi
