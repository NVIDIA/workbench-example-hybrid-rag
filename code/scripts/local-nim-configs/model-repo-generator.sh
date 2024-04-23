# This script generates the model repo for running the NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

##### TODO: If using a different model, adjust the docker command as needed. #####
echo "Logging into NGC. "
echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
echo "Running Model Repo Generator..."
docker run --rm --gpus=0 -v ${LOCAL_NIM_HOME}/model-store:/model-store -v ${LOCAL_NIM_HOME}/mistral-example.yaml:/MISTRAL-7b-INSTRUCT-1.yaml:ro -v ${LOCAL_NIM_HOME}/model-downloads/Mistral-7B-Instruct-v0.1:/model-downloads/mistral-7b-instruct-v0.1:ro nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 model_repo_generator llm --verbose --yaml_config_file=/MISTRAL-7b-INSTRUCT-1.yaml
echo "Model repo generation process finished."

exit 0
