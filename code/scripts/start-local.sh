# This script spins up the local TGI Inference Server. HF Model ID is a required parameters, quantization level is optional

pkill -SIGINT -f "^text-generation-server download-weights" & 
pkill -SIGINT -f '^text-generation-launcher' & 
pkill -SIGINT -f 'text-generation' & 

sleep 1

CUDA_MEMORY_FRACTION=0.85 # adjust the percentage of the GPU being used. Don't go above 0.95

if [ "$2" = "none" ]
then
    text-generation-launcher --model-id $1 --cuda-memory-fraction $CUDA_MEMORY_FRACTION --max-input-length 4000 --max-total-tokens 5000 --port 9090 --trust-remote-code &
else
    text-generation-launcher --model-id $1 --cuda-memory-fraction $CUDA_MEMORY_FRACTION --max-input-length 4000 --max-total-tokens 5000 --quantize $2 --port 9090 --trust-remote-code &
fi

# Wait for service to be reachable.
ATTEMPTS=0
MAX_ATTEMPTS=20

while [ $(curl -o /dev/null -s -w "%{http_code}" "http://localhost:9090/info") -ne 200 ]; 
do 
  ATTEMPTS=$(($ATTEMPTS+1))
  if [ ${ATTEMPTS} -eq ${MAX_ATTEMPTS} ]
  then
    echo "Max attempts reached: $MAX_ATTEMPTS. Server may have timed out. Did you spell the model name correctly? Stop the server and try again. "
    exit 1
  fi
  
  echo "Polling inference server. Awaiting status 200; trying again in 5s. "
  sleep 5
done 

echo "Service reachable. Happy chatting!"
exit 0