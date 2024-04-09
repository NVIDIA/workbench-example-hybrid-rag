# This script runs the NeMo Inference Microservice container locally on DOCKER-enabled systems. 

if [ -x "$(command -v docker)" ]; then
    echo "Docker detected. Proceeding with preflight checklist. "
else
    echo "Docker is not detected! Local NIM is currently supported on Docker runtimes only. "
    exit 1
fi

# Preflight checklist
if [[ -z "${NGC_CLI_API_KEY}" ]]; then
  echo "Missing config: the user has not configured their NGC_CLI_API_KEY as a project secret. Can't pull the NIM container!"
  exit 1
elif [[ -z "${DOCKER_HOST}" ]]; then
  echo "Missing config: the user has not configured their DOCKER_HOST as an env variable. See README for details. "
  exit 1
elif [[ -z "${LOCAL_NIM_MODEL_STORE}" ]]; then
  echo "Missing config: the user has not configured their LOCAL_NIM_MODEL_STORE as an env variable. See README for details. "
  exit 1
elif [ ! -d "/opt/host-run" ]; then
  echo "Missing config: could not find /opt/host-run. Docker socket mount appears unconfigured. See README for details. "
  exit 1
else
  echo "Configs look good. Logging into NGC now. "
  echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
fi

echo "Attempting to run the NIM Container. "

# Run the microservice
docker run --gpus '"device=0"' --shm-size=8G --rm -d --name local_nim --network workbench -v $LOCAL_NIM_MODEL_STORE:/model-store -p 9999:9999 -p 9998:9998 nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 nemollm_inference_ms --model $1 --openai_port="9999" --nemo_port="9998" --num_gpus=1 

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

echo "Service reachable."
exit 0
