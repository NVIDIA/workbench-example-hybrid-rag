# This script runs the NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

echo "Logging into NGC. "
echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin

echo "Attempting to run the NIM Container. "

# Run the microservice
docker run --gpus '"device=0"' --shm-size=8G --rm -d --name local_nim --network workbench -v $LOCAL_NIM_HOME/model-store:/model-store -p 9999:9999 -p 9998:9998 nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 nemollm_inference_ms --model $1 --openai_port="9999" --nemo_port="9998" --num_gpus=1 

# Wait for service to be reachable.
ATTEMPTS=0
MAX_ATTEMPTS=10

while [ "$(curl -s -o /dev/null -w "%{http_code}" -i 'POST' 'http://local_nim:9999/v1/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d "{  \"model\": \"$1\", \"prompt\": \"hello world\", \"max_tokens\": 1024, \"temperature\": 0.7,\"n\": 1, \"stream\": false, \"stop\": \"string\", \"frequency_penalty\": 0.0 }" | tail -c 3)" != "200" ]; 
do 
  ATTEMPTS=$(($ATTEMPTS+1))
  if [ ${ATTEMPTS} -eq ${MAX_ATTEMPTS} ]
  then
    echo "Max attempts reached: $MAX_ATTEMPTS. Server may have timed out. Did you spell the model name correctly? Stop the server and try again. "
    exit 1
  fi
  
  echo "Polling microservice. Awaiting status 200; trying again in 10s. "
  sleep 10
done 

echo "Service reachable. Happy chatting!"
exit 0
