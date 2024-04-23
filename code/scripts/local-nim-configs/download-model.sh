# This script downloads the model weights for running a NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

##### TODO: If using a different model, swap/replace the .yaml file to the desired model. #####
echo "Copying YAML file to host"
cp /project/code/scripts/local-nim-configs/mistral-example.yaml /mnt/tmp/mistral-example.yaml   
echo "Copied YAML file to host"

echo "Creating model-downloads and model-store directories on host"
cd /mnt/tmp && mkdir model-downloads
cd /mnt/tmp && mkdir model-store

##### TODO: If using a different model, swap/replace the URL to the desired Hugging Face model. #####
echo "Downloading model weights from HF"
cd /mnt/tmp/model-downloads
git lfs clone https://${HUGGING_FACE_HUB_USERNAME}:${HUGGING_FACE_HUB_TOKEN}@huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
echo "Model weights downloaded"

exit 0