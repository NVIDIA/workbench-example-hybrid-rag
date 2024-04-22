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
elif [[ -z "${LOCAL_NIM_HOME}" ]]; then
  echo "Missing config: the user has not configured their LOCAL_NIM_HOME as an env variable. See README for details. "
  exit 1
elif [ ! -d "/opt/host-run" ]; then
  echo "Missing config: could not find /opt/host-run. Docker socket mount appears unconfigured. See README for details. "
  exit 1
else
  echo "Configs look good. Moving to the next step. "
fi

##### TODO 1: If using a different model, swap/replace the .yaml file to the desired model. #####
echo "Copying YAML file to host"
cp /project/code/scripts/local-nim-configs/mistral-example.yaml /mnt/tmp/mistral-example.yaml   
echo "Copied YAML file to host"

echo "Creating model-downloads and model-store directories on host"
cd /mnt/tmp && mkdir model-downloads
cd /mnt/tmp && mkdir model-store

##### TODO 2: If using a different model, swap/replace the URL to the desired Hugging Face model. #####
echo "Downloading model weights from HF"
cd /mnt/tmp/model-downloads
git lfs clone https://${HUGGING_FACE_HUB_USERNAME}:${HUGGING_FACE_HUB_TOKEN}@huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
echo "Model weights downloaded"

##### TODO 3: If using a different model, adjust the docker command as needed. #####
echo "Logging into NGC. "
echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
echo "Running Model Repo Generator..."
docker run --rm -ti --gpus=0 -v ${LOCAL_NIM_HOME}/model-store:/model-store -v ${LOCAL_NIM_HOME}/mistral-example.yaml:/MISTRAL-7b-INSTRUCT-1.yaml:ro -v ${LOCAL_NIM_HOME}/model-downloads/Mistral-7B-Instruct-v0.1:/model-downloads/mistral-7b-instruct-v0.1:ro nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 model_repo_generator llm --verbose --yaml_config_file=/MISTRAL-7b-INSTRUCT-1.yaml
echo "Model repo generation process finished."

exit 0
