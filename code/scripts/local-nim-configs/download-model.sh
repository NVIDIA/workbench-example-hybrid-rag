# This script downloads the model weights for running a NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

##### TODO: If using a different model, swap/replace the .yaml file to the desired model. #####
echo "Copying YAML file to host"
cp /project/code/scripts/local-nim-configs/mistral-example.yaml /mnt/host-home/mistral-example.yaml   
echo "Copied YAML file to host"

echo "Creating model-downloads and model-store directories on host"
cd /mnt/host-home 
mkdir model-downloads
mkdir model-store
chmod -R 777 model-downloads
chmod -R 777 model-store
cd -

##### TODO: If using a different model, swap/replace the URL to the desired Hugging Face model. #####
echo "Downloading model weights from HF"
cd /mnt/host-home/model-downloads
git lfs clone https://${HUGGING_FACE_HUB_USERNAME}:${HUGGING_FACE_HUB_TOKEN}@huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
echo "Model weights downloaded"

exit 0
