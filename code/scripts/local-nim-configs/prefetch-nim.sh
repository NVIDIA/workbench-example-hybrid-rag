# This script generates the model repo for running the NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

##### TODO: If using a different model, adjust the docker command as needed. #####
echo "Logging into NGC. "
echo $NGC_CLI_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
echo "Running NIM pre-fetch step..."
docker pull $1
echo "NIM pre-fetch process finished."

exit 0
