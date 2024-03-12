#!/bin/bash

# This script runs a health check for the API server and the Milvus vector store

# List of URLs to check
URLS=("http://localhost:8000/health" "http://localhost:19530/v1/vector/collections")

for url in "${URLS[@]}"; do
    # Curl each URL, only outputting the HTTP status code
    status=$(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "$url")
    
    # Check if the status is not 200
    if [[ $status -ne 200 ]]; then
        echo "Error: $url returned HTTP code $status"
        exit 1
    fi
done

# If loop completes without exiting, all URLs returned 200
echo "All URLs returned HTTP code 200"
exit 0