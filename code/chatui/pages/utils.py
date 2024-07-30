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

import functools
import logging
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

import gradio as gr
import json
import shutil
import os
import subprocess
import time
import torch
import tiktoken
import fnmatch

from chatui import assets, chat_client

### Helper Functions used by the application. ### 

def upload_file(files: List[Path], client: chat_client.ChatClient) -> List[str]:
    """
    Use the client to upload a document to the vector database.
    
    Parameters: 
        files (List[Path]): List of filepaths to the files being uploaded
        client (chat_client.ChatClient): Chat client used for uploading files
    
    Returns:
        file_paths (List[str]): List of file names
    """
    file_paths = [file.name for file in files]
    client.upload_documents(file_paths)
    return file_paths

def inference_to_config(gradio: str) -> str:
    """
    Helper function to convert displayed inference mode string to a backend-readable string.
    
    Parameters: 
        gradio (str): Rendered inference mode string on frontend.
    
    Returns:
        (str): Backend-readable inference mode string.
    """
    if gradio == "Local System": 
        return "local"
    elif gradio == "Cloud Endpoint": 
        return "cloud"
    elif gradio == "Self-Hosted Microservice": 
        return "microservice"
    else:
        return gradio

def cloud_to_config(cloud: str) -> str:
    """
    Helper function to convert rendered cloud model string to a backend-readable endpoint.
    
    Parameters: 
        cloud (str): Rendered cloud model string on frontend.
    
    Returns:
        (str): Backend-readable cloud model endpoint.
    """
    if cloud == "Llama3 ChatQA-1.5 8B": 
        return "nvidia/llama3-chatqa-1.5-8b"
    elif cloud == "Llama3 ChatQA-1.5 70B": 
        return "nvidia/llama3-chatqa-1.5-70b"
    elif cloud == "Nemotron-4 340B Instruct": 
        return "nvidia/nemotron-4-340b-instruct"
    elif cloud == "Mistral-NeMo 12B Instruct": 
        return "nv-mistralai/mistral-nemo-12b-instruct"
    elif cloud == "Mistral 7B Instruct v0.2": 
        return "mistralai/mistral-7b-instruct-v0.2"
    elif cloud == "Mistral 7B Instruct v0.3": 
        return "mistralai/mistral-7b-instruct-v0.3"
    elif cloud == "Mistral Large": 
        return "mistralai/mistral-large"
    elif cloud == "Mixtral 8x7B Instruct v0.1": 
        return "mistralai/mixtral-8x7b-instruct-v0.1"
    elif cloud == "Mixtral 8x22B Instruct v0.1": 
        return "mistralai/mixtral-8x22b-instruct-v0.1"
    elif cloud == "Codestral 22B Instruct v0.1": 
        return "mistralai/codestral-22b-instruct-v0.1"
    elif cloud == "Llama 2 70B": 
        return "meta/llama2-70b"
    elif cloud == "Llama 3 8B": 
        return "meta/llama3-8b-instruct"
    elif cloud == "Llama 3 70B": 
        return "meta/llama3-70b-instruct"
    elif cloud == "Llama 3.1 8B": 
        return "meta/llama-3.1-8b-instruct"
    elif cloud == "Llama 3.1 70B": 
        return "meta/llama-3.1-70b-instruct"
    elif cloud == "Llama 3.1 405B": 
        return "meta/llama-3.1-405b-instruct"
    elif cloud == "Gemma 2B": 
        return "google/gemma-2b"
    elif cloud == "Gemma 7B": 
        return "google/gemma-7b"
    elif cloud == "Code Gemma 7B": 
        return "google/codegemma-7b"
    elif cloud == "Phi-3 Mini (4k)": 
        return "microsoft/phi-3-mini-4k-instruct"
    elif cloud == "Phi-3 Mini (128k)": 
        return "microsoft/phi-3-mini-128k-instruct"
    elif cloud == "Phi-3 Small (8k)": 
        return "microsoft/phi-3-small-8k-instruct"
    elif cloud == "Phi-3 Small (128k)": 
        return "microsoft/phi-3-small-128k-instruct"
    elif cloud == "Phi-3 Medium (4k)": 
        return "microsoft/phi-3-medium-4k-instruct"
    elif cloud == "Arctic": 
        return "snowflake/arctic"
    elif cloud == "Granite 8B Code": 
        return "ibm/granite-8b-code-instruct"
    elif cloud == "Granite 34B Code": 
        return "ibm/granite-34b-code-instruct"
    elif cloud == "Solar 10.7B Instruct": 
        return "upstage/solar-10.7b-instruct"
    else:
        return "mistralai/mistral-7b-instruct-v0.2"

def quant_to_config(quant: str) -> str:
    """
    Helper function to convert rendered quantization string to a backend-readable string.
    
    Parameters: 
        quant (str): Rendered quantization string on frontend.
    
    Returns:
        (str): Backend-readable quantization string.
    """
    if quant == "None": 
        return "none"
    elif quant == "8-Bit": 
        return "bitsandbytes"
    elif quant == "4-Bit": 
        return "bitsandbytes-nf4"
    else:
        return "none"

def preset_quantization() -> str:
    """
    Helper function to introspect the system and preset the recommended quantization level.
    
    Parameters: 
        None
    
    Returns:
        (str): quantization level to be rendered on the frontend application.
    """
    inf_mem = 0
    for i in range(torch.cuda.device_count()):
        inf_mem += torch.cuda.get_device_properties(i).total_memory
    gb = inf_mem/(2**30)
    
    if gb >= 40:
        return "None"
    elif gb >= 24:
        return "8-Bit"
    else:
        return "4-Bit"

def preset_max_tokens() -> str:
    """
    Helper function to introspect the system and preset the range of max new tokens to generate.
    
    Parameters: 
        None
    
    Returns:
        (int): max new tokens to generate to be rendered on the frontend application.
    """
    inf_mem = 0
    for i in range(torch.cuda.device_count()):
        inf_mem += torch.cuda.get_device_properties(i).total_memory
    gb = inf_mem/(2**30)
    
    if gb >= 40:
        return 512, 2048
    elif gb >= 24:
        return 512, 1024
    else:
        return 256, 512

def clear_knowledge_base() -> bool:
    """
    Helper function to run a script to clear out the vector database.
    
    Parameters: 
        None
    
    Returns:
        (bool): True if completed with exit code 0, else False.
    """
    rc = subprocess.call("/bin/bash /project/code/scripts/clear-docs.sh", shell=True)
    return True if rc == 0 else False

def start_local_server(local_model_id: str, local_model_quantize: str) -> bool:
    """
    Helper function to run a script to start the local TGI inference server.
    
    Parameters: 
        local_model_id (str): The model name selected by the user
        local_model_quantize (str): The quantization level selected by the user
    
    Returns:
        (bool): True if completed with exit code 0, else False.
    """
    rc = subprocess.call("/bin/bash /project/code/scripts/start-local.sh " + local_model_id + " " + local_model_quantize, shell=True)
    return True if rc == 0 else False

def stop_local_server() -> bool:
    """
    Helper function to run a script to stop the local TGI inference server.
    
    Parameters: 
        None
    
    Returns:
        (bool): True if completed with exit code 0, else False.
    """
    rc = subprocess.call("/bin/bash /project/code/scripts/stop-local.sh", shell=True)
    return True if rc == 0 else False

def nim_extract_model(input_string: str):
    """
    A helper function to convert a container "registry/image:tag" into a model name for NIMs

    Parameters: 
        input_string: full container URL, eg. "nvcr.io/nim/meta/llama3-8b-instruct:latest"
    
    Returns:
        substring: Name of the model for OpenAI API spec, eg. "meta/llama3-8b-instruct"
    """
    # Split the string by forward slashes
    parts = input_string.split('/')
    
    # If there are less than 3 parts, return the NIM playbook default model name
    if len(parts) < 3:
        return "meta/llama3-8b-instruct"
    
    # Get the substring after the second-to-last forward slash
    substring = parts[-2] + '/' + parts[-1]
    
    # If a colon exists, split the substring at the first colon
    if ':' in substring:
        substring = substring.split(':')[0]
    
    return substring

def get_initial_metrics(metrics_history, 
                        response_num, 
                        inference_mode, 
                        nvcf_model_id, 
                        local_model_id, 
                        nim_local_model_id, 
                        is_local_nim, 
                        nim_model_id, 
                        retrieval_ftime, 
                        ttft):

    """
    A helper function to generate the initial metrics as the response is being streamed

    Parameters: 
        metrics_history: dict of metrics previously calculated
        response_num: number of responses generated so far
        inference_mode: user selected inference mode 
        nvcf_model_id: user selected cloud model endpoint
        local_model_id: user selected local model id
        nim_local_model_id: user selected local nim model id
        is_local_nim: bool for if local nim is currently selected by user
        nim_model_id: user selected remote nim model id
        retrieval_ftime: retrieval time in ms
        ttft: time to first token in ms
    
    Returns:
        (Dict): Updated dict of calculated metrics
    """

    return metrics_history.update({str(response_num): {"inference_mode": inference_to_config(inference_mode),
                                                       "model": nvcf_model_id if inference_to_config(inference_mode)=="cloud" else 
                                                                (local_model_id if inference_to_config(inference_mode)=="local" else 
                                                                (nim_extract_model(nim_local_model_id) if inference_to_config(inference_mode) 
                                                                 and is_local_nim else nim_model_id)),
                                                       "Retrieval time": "N/A" if len(retrieval_ftime) == 0 else retrieval_ftime + "ms",
                                                       "Time to First Token (TTFT)": ttft + "ms"}})

def get_final_metrics(e2e_ftime, 
                      e2e_stime, 
                      ttft, 
                      retrieval_ftime, 
                      chunks):

    """
    A helper function to generate the initial metrics as the response is being streamed

    Parameters: 
        metrics_history: dict of metrics previously calculated
        e2e_ftime: time, in seconds, or the end-to-end query
        ttft: time to first token
        retrieval_ftime: retrieval time in ms
        tokens: number of tokens generated in response
        response_num: current response number
    
    Returns:
        (Dict): Updated dict of calculated metrics
    """

    e2e_ftime = str((e2e_ftime - e2e_stime) * 1000).split('.', 1)[0]
    gen_time = int(e2e_ftime) - int(ttft) if len(retrieval_ftime) == 0 else int(e2e_ftime) - int(ttft) - int(retrieval_ftime)
    tokens = len(tiktoken.get_encoding('cl100k_base').encode(chunks))
    return str(gen_time), e2e_ftime, str(tokens), str(round(tokens / (gen_time / 1000), 1)), str(round((gen_time / tokens), 1))
    