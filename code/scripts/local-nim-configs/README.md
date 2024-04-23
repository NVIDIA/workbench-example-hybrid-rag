**Note**: This README file is meant as a _supplement_ to the primary README file for the project [here](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md). 

To swap out the default Mistral-7B-Instruct-v0.1 model, complete the following steps. 

1. Create a .yaml configuration file for your desired model. Use ``mistral-example.yaml`` as reference, as well as the sample configs for Llama and Nemotron provided on the NIM documentation. 

2. In ``download-model.sh``,

   * Adjust the copy command to copy the correct .yaml config file you configured for step 1.
   * Adjust the Hugging Face clone link to pull the correct model weights for your desired model. 

3. In ``model-repo-generator.sh``, adjust the docker command to reflect your updated model. Namely, adjust the .yaml filename and the model name.

4. Save your changes. When selecting **Start Microservice**, be sure to first specify your updated model name instead of the default ``Mistral-7B-Instruct-v0.1``. 
