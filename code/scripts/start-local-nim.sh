# This script runs the NeMo Inference Microservice container locally on DOCKER-enabled systems. 

# Login to NGC, if needed
echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Run the microservice
docker run --gpus '"device=0"' --shm-size=8G --rm -d --name local_nim --network workbench -v $LOCAL_NIM_MODEL_STORE:/model-store -p 9999:9999 -p 9998:9998 nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:24.01 nemollm_inference_ms --model $1 --openai_port="9999" --nemo_port="9998" --num_gpus=1 

sleep 50
exit 0
