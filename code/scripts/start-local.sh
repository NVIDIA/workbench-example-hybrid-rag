# This script spins up the local TGI Inference Server. HF Model ID is a required parameters, quantization level is optional

pkill -SIGINT -f "^text-generation-server download-weights" & 
pkill -SIGINT -f '^text-generation-launcher' & 
pkill -SIGINT -f 'text-generation' & 

sleep 1

CUDA_MEMORY_FRACTION=0.75 # adjust the percentage of the GPU being used. Don't go above 0.95

if [ "$2" = "none" ]
then
    text-generation-launcher --model-id $1 --cuda-memory-fraction $CUDA_MEMORY_FRACTION --port 9090 &
else
    text-generation-launcher --model-id $1 --cuda-memory-fraction $CUDA_MEMORY_FRACTION --quantize $2 --port 9090 &
fi

sleep 30 # Model warm-up

URLS=("http://localhost:9090/info")

for url in "${URLS[@]}"; do
    # Curl each URL, only outputting the HTTP status code
    status=$(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "$url")
    
    # Check if the status is not 200
    if [[ $status -ne 200 ]]; then
        echo "Error: $url returned HTTP code $status"
        exit 1
    fi
done

sleep 1
exit 0