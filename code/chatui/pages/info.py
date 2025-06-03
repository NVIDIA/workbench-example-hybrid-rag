# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Markdown used to render certain documentation on the gradio application. ###

setup = """
Welcome to the Hybrid RAG example project for NVIDIA AI Workbench! \n\nTo get started, click the following button to set up the backend API server and vector database. This is a one-time process and may take a few moments to complete.
"""

update_kb_info = """
<br> 
Upload your text files here. This will embed them in the vector database, and they will persist as potential context for the model until you clear the database. Careful, clearing the database is irreversible!
"""

inf_mode_info = "To use a CLOUD endpoint for inference, select the desired model before making a query."

local_info = """
First, select the desired model and quantization level. You can optionally filter the model list by gated vs ungated models. Then load the model. This will either download it or load it from cache. The download may take a few minutes depending on your network. 

Once the model is loaded, start the Inference Server. It takes ~40s to warm up in most cases. Ensure you have enough GPU VRAM to run a model locally or you may see OOM errors when starting the inference server. When the server is started, chat with the model using the text input on the left.
"""

local_prereqs = """
* A ``HUGGING_FACE_HUB_TOKEN`` project secret is required for gated models. See [Tutorial 1](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-1-using-a-local-gpu). 
* If using any of the following gated models, verify "You have been granted access to this model" appears on the model card(s):
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

local_trouble = """
* Ensure you have stopped any local processes also running on the system GPU(s). Otherwise, you may run into OOM errors running on the local inference server. 
* Your Hugging Face key may be missing and/or lack permissions for certain models. Ensure you see a "You have been granted access to this model" for each page: 
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

cloud_info = """
This method uses NVCF API Endpoints from the NVIDIA API Catalog. Select a desired model family and model from the dropdown. You may then query the model using the text input on the left.
"""

cloud_prereqs = """
* A ``NVIDIA_API_KEY`` project secret is required. See the [Quickstart](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-using-a-cloud-endpoint). 
    * Generate the key [here](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2) by clicking "Get API Key". Log in with [NGC credentials](https://ngc.nvidia.com/signin).
"""

cloud_trouble = """
* Ensure your NVCF run key is correct and configured properly in the AI Workbench. 
"""

nim_info = """
This method uses a [NIM container](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama3-8b-instruct/tags) that you may choose to self-host on your own infra of choice via the Compose feature in AI Workbench. Check out the NIM [docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) for details. Users can also try 3rd party services supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Input the desired microservice name if running locally or IP/hostname if running remotely, optional port number, and model name. Then, start conversing using the text input on the left.
"""

nim_prereqs = """
* (Remote) Set up a NIM running on another system ([docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)). Alternatively, you may set up a 3rd party supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Ensure your service is running and reachable. See [Tutorial 2](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-2-using-a-remote-microservice). 
* (Local) Start the Compose service in the AI Workbench window. Wait a few minutes to ensure your service is running and reachable. You can read and follow the additional prereqs and configurations in [Tutorial 3](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-3-using-a-local-microservice). 
"""

nim_trouble = """
* Send a curl request to your microservice to ensure it is running and reachable. NIM docs [here](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).
* If any other processes are running on the local GPU(s), you may run into memory issues when also running the NIM locally. Stop the other processes. 
"""

num_token_label = """
The maximum number of tokens that can be generated in the completion.
"""

temp_label = """
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
"""

top_p_label = """
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
"""

freq_pen_label = """
Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
"""

pres_pen_label = """
Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
"""