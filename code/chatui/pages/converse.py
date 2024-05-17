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

"""This module contains the chatui gui for having a conversation."""
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

from chatui import assets, chat_client

_LOGGER = logging.getLogger(__name__)
PATH = "/"
TITLE = "Hybrid RAG: Chat UI"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

_LOCAL_CSS = """

#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j {
    color: #76b900;
}

"""

update_kb_info = """
<br> 
Upload your text files here. This will embed them in the vector database, and they will be the context for the model until you clear the database. Careful, clearing the database is irreversible!
"""

inf_mode_info = "To use a CLOUD endpoint for inference, select the desired model before making a query."

local_info = """
First, select the desired model and quantization level. Then load the model. This will either download it or load it from cache. The download may take a few minutes depending on your network. 

Once the model is loaded, start the Inference Server. It takes ~30s to warm up. Ensure you have enough GPU VRAM to run a model locally or you may see OOM errors when starting the inference server. When the server is started, chat with the model using the text input on the left.
"""

local_prereqs = """
* A ``HUGGING_FACE_HUB_TOKEN`` project secret is recommended. See [Tutorial 1](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-1-using-a-local-gpu). 
* If using any of the following models, verify "You have been granted access to this model" appears on the model card(s):
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

local_trouble = """
* Ensure you have stopped any local processes also running on the system GPU(s). Otherwise, you may run into OOM errors running on the local inference server. 
* Your Hugging Face key may be missing permissions for certain models. Ensure you see a "You have been granted access to this model" for each page: 
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

cloud_info = """
This method uses NVCF API Endpoints from the NVIDIA API Catalog. Select a desired model family and model from the dropdown. You may then query the model using the text input on the left.
"""

cloud_prereqs = """
* A ``NVCF_RUN_KEY`` project secret is required. See the [Quickstart](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-using-a-cloud-endpoint). 
    * Generate the key [here](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2) by clicking "Get API Key". Log in with [NGC credentials](https://ngc.nvidia.com/signin).
"""

cloud_trouble = """
* Ensure your NVCF run key is correct and configured properly in the AI Workbench. 
"""

nim_info = """
This method uses a [NIM container](https://developer.nvidia.com/nemo-microservices-early-access) that you may choose to self-host on your own infra of choice. Check out the NIM [docs](https://developer.nvidia.com/docs/nemo-microservices/nmi_playbook.html) for details (you may need to access it via NGC). Users can also try 3rd party services supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Input the desired microservice IP, optional port number, and model name under the Remote Microservice option. Then, start conversing using the text input on the left.

For AI Workbench on DOCKER users only, you may also choose to spin up a NIM instance running locally on the system by expanding the "Local" Microservice option; ensure any other local GPU processes has been stopped first to avoid issues with memory. Mistral-7b-instruct-v0.1 is provided as a default flow; to swap models, adjustments in the codebase are required. 

To run mistral-7b-instruct-v0.1, leave the model name field as default and generate the model repository (can take several minutes); then press "Start Microservice" and begin conversing. Use the project README, NIM documentation, and this project's codebase as resources for implementation details. 
"""

nim_prereqs = """
* [NIM Early Access](https://developer.nvidia.com/nemo-microservices-early-access) and [NIM documentation](https://developer.nvidia.com/docs/nemo-microservices/index.html). Alternatively, you may set up a 3rd party supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building)
* (Remote) Ensure your microservice is set up and running properly. See [Tutorial 2](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-2-using-a-remote-microservice). 
* (Local) AI Workbench running on DOCKER is required for the LOCAL NIM option. Read and follow the additional prereqs and configurations in [Tutorial 3](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-3-using-a-local-microservice). 
"""

nim_trouble = """
* Send a curl request to your microservice to ensure it is running and reachable. NIM docs [here](https://developer.nvidia.com/docs/nemo-microservices/nmi_playbook.html).
* AI Workbench running on a Docker runtime is required for the LOCAL NIM option. Otherwise, set up the self-hosted NIM to be used remotely. 
* If running the local NIM option, ensure you have set up the proper project configurations according to this project's README. Unlike the other inference modes, these are not preconfigured. 
* If any other processes are running on the local GPU(s), you may run into memory issues when also running the NIM locally. Stop the other processes. 
"""

def upload_file(files: List[Path], client: chat_client.ChatClient) -> List[str]:
    """Use the client to upload a document to the vector database."""
    file_paths = [file.name for file in files]
    client.upload_documents(file_paths)
    return file_paths

def inference_to_config(gradio: str) -> str:
    if gradio == "Local System": 
        return "local"
    elif gradio == "Cloud Endpoint": 
        return "cloud"
    elif gradio == "Self-Hosted Microservice": 
        return "microservice"
    else:
        return gradio

def cloud_to_config(cloud: str) -> str:
    if cloud == "Mistral 7B": 
        return "mistralai/mistral-7b-instruct-v0.2"
    elif cloud == "Mistral Large": 
        return "mistralai/mistral-large"
    elif cloud == "Mixtral 8x7B": 
        return "mistralai/mixtral-8x7b-instruct-v0.1"
    elif cloud == "Mixtral 8x22B": 
        return "mistralai/mixtral-8x22b-instruct-v0.1"
    elif cloud == "Llama 2 70B": 
        return "meta/llama2-70b"
    elif cloud == "Llama 3 8B": 
        return "meta/llama3-8b-instruct"
    elif cloud == "Llama 3 70B": 
        return "meta/llama3-70b-instruct"
    elif cloud == "Gemma 2B": 
        return "google/gemma-2b"
    elif cloud == "Gemma 7B": 
        return "google/gemma-7b"
    elif cloud == "Code Gemma 7B": 
        return "google/codegemma-7b"
    elif cloud == "Phi-3 Mini (128k)": 
        return "microsoft/phi-3-mini-128k-instruct"
    elif cloud == "Arctic": 
        return "snowflake/arctic"
    else:
        return "mistralai/mistral-7b-instruct-v0.2"

def quant_to_config(quant: str) -> str:
    if quant == "None": 
        return "none"
    elif quant == "8-Bit": 
        return "bitsandbytes"
    elif quant == "4-Bit": 
        return "bitsandbytes-nf4"
    else:
        return "none"

def preset_quantization() -> str:
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

def clear_knowledge_base() -> bool:
    rc = subprocess.call("/bin/bash /project/code/scripts/clear-docs.sh", shell=True)
    return True if rc == 0 else False

def start_local_server(local_model_id: str, local_model_quantize: str) -> bool:
    rc = subprocess.call("/bin/bash /project/code/scripts/start-local.sh " + local_model_id + " " + local_model_quantize, shell=True)
    return True if rc == 0 else False

def stop_local_server() -> bool:
    rc = subprocess.call("/bin/bash /project/code/scripts/stop-local.sh", shell=True)
    return True if rc == 0 else False

def start_local_nim() -> bool:
    rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/start-local-nim.sh ", shell=True)
    return True if rc == 0 else False

def stop_local_nim() -> bool:
    rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/stop-local-nim.sh", shell=True)
    return True if rc == 0 else False

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """Build the gradio page to be mounted in the frame."""
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        # create the page header
        gr.Markdown(f"# {TITLE}")

        # State
        which_nim_tab = gr.State(0)
        is_local_nim = gr.State(False)
        vdb_active = gr.State(False)
        metrics_history = gr.State({})

        # chat logs
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=350):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=350):
                        chatbot = gr.Chatbot(show_label=False)
                    context = gr.JSON(
                        scale=1,
                        label="Vector Database Context",
                        visible=False,
                        elem_id="contextbox",
                    )
                    metrics = gr.JSON(
                        scale=1,
                        label="Metrics",
                        visible=False,
                        elem_id="contextbox",
                    )
                with gr.Row(equal_height=True):
                    num_token_slider = gr.Slider(0, 512, value=256, label="Max Tokens in Response", interactive=True)
                    temp_slider = gr.Slider(0, 1, value=0.7, label="Temperature", interactive=True)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=200):
                        msg = gr.Textbox(
                            show_label=False,
                            lines=3,
                            placeholder="Enter text and press SUBMIT",
                            container=False,
                            interactive=True,
                        )
                    with gr.Column(scale=1, min_width=100):
                        kb_checkbox = gr.CheckboxGroup(
                            ["Toggle to use Vector Database"], label="Vector Database", info="Supply your uploaded documents to the chatbot"
                        )
                with gr.Row():
                    submit_btn = gr.Button(value="[NOT READY] Submit", interactive=False)
                    _ = gr.ClearButton([msg, chatbot, metrics, metrics_history], value="Clear history")
                    mtx_show = gr.Button(value="Show Metrics")
                    mtx_hide = gr.Button(value="Hide Metrics", visible=False)
                    ctx_show = gr.Button(value="Show Context")
                    ctx_hide = gr.Button(value="Hide Context", visible=False)
                                    
            with gr.Column(scale=2, min_width=350, visible=True) as settings_column:
                with gr.Tabs(selected=0):
                    with gr.TabItem("Initial Setup", id=0, interactive=False, visible=True) as setup_settings:
                        gr.Markdown("<br> ")
                        gr.Markdown("Welcome to the Hybrid RAG example project for NVIDIA AI Workbench! \n\nTo get started, click the following button to set up the backend API server and vector database. This is a one-time process and may take a few moments to complete.")
                        rag_start_button = gr.Button(value="Set Up RAG Backend", variant="primary")
                        gr.Markdown("<br> ")
                    with gr.TabItem("Inference Settings", id=1, interactive=False, visible=True) as inf_settings:
                
                        inference_mode = gr.Radio(["Local System", "Cloud Endpoint", "Self-Hosted Microservice"], label="Inference Mode", info=inf_mode_info, value="Cloud Endpoint")
        
                        with gr.Tabs(selected=1) as tabs:
                            with gr.TabItem("Local System", id=0, interactive=False, visible=False) as local:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(local_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(local_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(local_trouble)
                                
                                local_model_id = gr.Dropdown(choices = ["nvidia/Llama3-ChatQA-1.5-8B", 
                                                                        "mistralai/Mistral-7B-Instruct-v0.1",
                                                                        "mistralai/Mistral-7B-Instruct-v0.2",
                                                                        "meta-llama/Llama-2-7b-chat-hf",
                                                                        "meta-llama/Meta-Llama-3-8B-Instruct"], 
                                                             value = "nvidia/Llama3-ChatQA-1.5-8B",
                                                             interactive = True,
                                                             label = "Select a model (or input your own).", 
                                                             allow_custom_value = True, 
                                                             elem_id="rag-inputs")
                                local_model_quantize = gr.Dropdown(choices = ["None",
                                                                              "8-Bit",
                                                                              "4-Bit"], 
                                                                   value = preset_quantization(),
                                                                   interactive = True,
                                                                   label = "Select model quantization.", 
                                                                   elem_id="rag-inputs")
                                
                                with gr.Row(equal_height=True):
                                    download_model = gr.Button(value="Load Model", size="sm")
                                    start_local_server = gr.Button(value="Start Server", interactive=False, size="sm")
                                    stop_local_server = gr.Button(value="Stop Server", interactive=False, size="sm")
                                    
                            with gr.TabItem("Cloud Endpoint", id=1, interactive=False, visible=False) as cloud:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(cloud_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(cloud_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(cloud_trouble)
                                
                                nvcf_model_family = gr.Dropdown(choices = ["Select", 
                                                                           "MistralAI", 
                                                                           "Meta", 
                                                                           "Google",
                                                                           "Microsoft", 
                                                                           "Snowflake"], 
                                                                value = "Select", 
                                                                interactive = True,
                                                                label = "Select a model family.", 
                                                                elem_id="rag-inputs")
                                nvcf_model_id = gr.Dropdown(choices = ["Select"], 
                                                            value = "Select",
                                                            interactive = True,
                                                            label = "Select a model.", 
                                                            visible = False,
                                                            elem_id="rag-inputs")
                            with gr.TabItem("Self-Hosted Microservice", id=2, interactive=False, visible=False) as microservice:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(nim_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(nim_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(nim_trouble)
        
                                with gr.Tabs(selected=0) as nim_tabs:
                                    with gr.TabItem("Remote", id=0) as remote_microservice:
                                        remote_nim_msg = gr.Markdown("<br />Enter the details below. Then start chatting!")
                                        with gr.Row(equal_height=True):
                                            nim_model_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                                       label = "Microservice Host", 
                                                       info = "IP Address running the microservice", 
                                                       elem_id="rag-inputs", scale=2)
                                            nim_model_port = gr.Textbox(placeholder = "9999", 
                                                       label = "Port", 
                                                       info = "Optional, (default: 9999)", 
                                                       elem_id="rag-inputs", scale=1)
                                        nim_model_id = gr.Textbox(placeholder = "llama-2-7b-chat", 
                                                   label = "Model running in microservice.", 
                                                   elem_id="rag-inputs")
                                    with gr.TabItem("Local", id=1) as local_microservice:
                                        gr.Markdown("<br />**Important**: For AI Workbench on DOCKER users only. Podman is unsupported!")
                                        gr.Markdown("This project provides an example for spinning up a local NIM running *mistral-7b-instruct-v0.1*. Open JupyterLab and adjust ``code/scripts/local-nim-configs/`` to bring your own custom models.")
                                        nim_local_model_id = gr.Textbox(value = "mistral-7b-instruct-v0.1", 
                                                   label = "Model running in microservice.", 
                                                   elem_id="rag-inputs")
                                        with gr.Row(equal_height=True):
                                            model_repo_generate = gr.Button(value="Generate Model Repo", size="sm")
                                            start_local_nim = gr.Button(value="Start Microservice", interactive=(True if os.path.isdir('/mnt/host-home/model-store') else False), size="sm")
                                            stop_local_nim = gr.Button(value="Stop Microservice", interactive=False, size="sm")
                                        
                    with gr.TabItem("Upload Documents Here", id=2, interactive=False, visible=True) as vdb_settings:
                        
                        gr.Markdown(update_kb_info)
                        
                        # file_output = gr.File(interactive=False, height=50)
                        file_output = gr.File(interactive=True, 
                                              show_label=False, 
                                              file_types=["text",
                                                          ".pdf",
                                                          ".html",
                                                          ".doc",
                                                          ".docx",
                                                          ".txt",
                                                          ".odt",
                                                          ".rtf",
                                                          ".tex"], 
                                              file_count="multiple")
        
                        with gr.Row():
                            clear_docs = gr.Button(value="Clear Database", interactive=False, size="sm") 

        # hide/show context
        def _toggle_info(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Show Context":
                out = [True, False, False, True, True, False]
            elif btn == "Hide Context":
                out = [False, False, True, False, True, False]
            elif btn == "Show Metrics":
                out = [False, True, True, False, False, True]
            elif btn == "Hide Metrics":
                out = [False, False, True, False, True, False]
            return {
                context: gr.update(visible=out[0]),
                metrics: gr.update(visible=out[1]),
                ctx_show: gr.update(visible=out[2]),
                ctx_hide: gr.update(visible=out[3]),
                mtx_show: gr.update(visible=out[4]),
                mtx_hide: gr.update(visible=out[5]),
            }

        ctx_show.click(_toggle_info, [ctx_show], [context, metrics, ctx_show, ctx_hide, mtx_show, mtx_hide])
        ctx_hide.click(_toggle_info, [ctx_hide], [context, metrics, ctx_show, ctx_hide, mtx_show, mtx_hide])
        mtx_show.click(_toggle_info, [mtx_show], [context, metrics, ctx_show, ctx_hide, mtx_show, mtx_hide])
        mtx_hide.click(_toggle_info, [mtx_hide], [context, metrics, ctx_show, ctx_hide, mtx_show, mtx_hide])

        def _toggle_model_download(btn: str, model: str, start: str, stop: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    download_model: gr.update(),
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                }
            else: 
                if btn == "Load Model":
                    progress(0.25, desc="Initializing Task")
                    time.sleep(0.75)
                    progress(0.5, desc="Downloading Model (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/download-local.sh " + model, shell=True)
                    if rc == 0:
                        msg = "Model Downloaded"
                        colors = "primary"
                        interactive = False
                        start_interactive = True if (start == "Start Server") else False
                        stop_interactive = True if (stop == "Stop Server") else False
                    else: 
                        msg = "Error, Try Again"
                        colors = "stop"
                        interactive = True
                        start_interactive = False
                        stop_interactive = False
                progress(0.75, desc="Cleaning Up")
                time.sleep(0.75)
                return {
                    download_model: gr.update(value=msg, variant=colors, interactive=interactive),
                    start_local_server: gr.update(interactive=start_interactive),
                    stop_local_server: gr.update(interactive=stop_interactive),
                }
        
        download_model.click(_toggle_model_download,
                             [download_model, local_model_id, start_local_server, stop_local_server], 
                             [download_model, start_local_server, stop_local_server, msg])

        def _toggle_model_select(model: str, start: str, stop: str) -> Dict[gr.component, Dict[Any, Any]]:
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
            return {
                download_model: gr.update(value="Load Model", 
                                          variant="secondary", 
                                          interactive=(False if start == "Server Started" else True)),
                start_local_server: gr.update(interactive=False),
                stop_local_server: gr.update(interactive=(False if stop == "Server Stopped" else True)),
            }
        
        local_model_id.change(_toggle_model_select,
                              [local_model_id, start_local_server, stop_local_server], 
                              [download_model, start_local_server, stop_local_server])

        def toggle_nvcf_family(family: str) -> Dict[gr.component, Dict[Any, Any]]:
            interactive = True
            submit_value = "Submit"
            msg_value = "Enter text and press SUBMIT"
            if family == "MistralAI":
                choices = ["Mistral 7B", "Mistral Large", "Mixtral 8x7B", "Mixtral 8x22B"]
                value = "Mistral 7B"
                visible = True
            elif family == "Meta":
                choices = ["Llama 2 70B", "Llama 3 8B", "Llama 3 70B"]
                value = "Llama 2 70B"
                visible = True
            elif family == "Google":
                choices = ["Gemma 2B", "Gemma 7B", "Code Gemma 7B"]
                value = "Gemma 2B"
                visible = True
            elif family == "Microsoft":
                choices = ["Phi-3 Mini (128k)"]
                value = "Phi-3 Mini (128k)"
                visible = True
            elif family == "Snowflake":
                choices = ["Arctic"]
                value = "Arctic"
                visible = True
            else:
                choices = ["Select"]
                value = "Select"
                visible = False
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Select a model OR Select a Different Inference Mode."
            return {
                nvcf_model_id: gr.update(choices=choices, value=value, visible=visible),
                submit_btn: gr.update(value=submit_value, interactive=interactive),
                msg: gr.update(interactive=True, 
                               placeholder=msg_value),
            }
        
        nvcf_model_family.change(toggle_nvcf_family,
                              [nvcf_model_family], 
                              [nvcf_model_id, submit_btn, msg])

        def _toggle_local_server(btn: str, model: str, quantize: str, download: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                    msg: gr.update(),
                    submit_btn: gr.update(),
                    download_model: gr.update(),
                }
            else: 
                if btn == "Start Server":
                    progress(0.2, desc="Initializing Task")
                    time.sleep(0.5)
                    progress(0.4, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
                    time.sleep(0.5)
                    progress(0.6, desc="Starting Inference Server (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/start-local.sh " 
                                              + model + " " + quant_to_config(quantize), shell=True)
                    if rc == 0:
                        out = ["Server Started", "Stop Server"]
                        colors = ["primary", "secondary"]
                        interactive = [False, True, True, False]
                    else: 
                        gr.Warning("You may be facing authentication or OOM issues. Check Troubleshooting for details.")
                        out = ["Internal Server Error, Try Again", "Stop Server"]
                        colors = ["stop", "secondary"]
                        interactive = [False, True, False, False]
                    progress(0.8, desc="Cleaning Up")
                    time.sleep(0.5)
                elif btn == "Stop Server":
                    progress(0.25, desc="Initializing")
                    time.sleep(0.5)
                    progress(0.5, desc="Stopping Server")
                    rc = subprocess.call("/bin/bash /project/code/scripts/stop-local.sh", shell=True)
                    if rc == 0:
                        out = ["Start Server", "Server Stopped"]
                        colors = ["secondary", "primary"]
                        interactive = [True, False, False, False if (download=="Model Downloaded") else True]
                    else: 
                        out = ["Start Server", "Internal Server Error, Try Again"]
                        colors = ["secondary", "stop"]
                        interactive = [True, False, True, False]
                    progress(0.75, desc="Cleaning Up")
                    time.sleep(0.5)
                return {
                    start_local_server: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                    stop_local_server: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                    msg: gr.update(interactive=True, 
                                   placeholder=("Enter text and press SUBMIT" if interactive[2] else "[NOT READY] Start the Local Inference Server OR Select a Different Inference Mode.")),
                    submit_btn: gr.update(value="Submit" if interactive[2] else "[NOT READY] Submit", interactive=interactive[2]),
                    download_model: gr.update(interactive=interactive[3]),
                }

        start_local_server.click(_toggle_local_server, 
                                 [start_local_server, local_model_id, local_model_quantize, download_model], 
                                 [start_local_server, stop_local_server, msg, submit_btn, download_model])
        stop_local_server.click(_toggle_local_server, 
                                 [stop_local_server, local_model_id, local_model_quantize, download_model], 
                                 [start_local_server, stop_local_server, msg, submit_btn, download_model])    

        def _toggle_nim_select(model: str, start: str, stop: str) -> Dict[gr.component, Dict[Any, Any]]:
            return {
                model_repo_generate: gr.update(value="Generate Model Repo", 
                                               variant="secondary", 
                                               interactive=(False if start == "Microservice Started" else True)),
                start_local_nim: gr.update(interactive=(True if start == "Start Microservice" and os.path.isdir('/mnt/host-home/model-store') else False)),
                stop_local_nim: gr.update(interactive=(True if start == "Microservice Started" else False)),
            }
        
        nim_local_model_id.change(_toggle_nim_select,
                              [nim_local_model_id, start_local_nim, stop_local_nim], 
                              [model_repo_generate, start_local_nim, stop_local_nim])

        def _toggle_model_repo_generate(btn: str, model: str, start: str, stop: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Generate Model Repo":
                progress(0.1, desc="Initializing Task")
                time.sleep(0.5)
                if len(model) == 0:
                    gr.Warning("Model name cannot be empty.")
                    msg = "Generate Model Repo"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        model_repo_generate: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                progress(0.2, desc="Checking user configs...")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/preflight.sh", shell=True)
                if rc != 0:
                    gr.Warning("You may have improper configurations set for this mode. Check the Output in the AI Workbench UI for details.")
                    msg = "Generate Model Repo"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        model_repo_generate: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                progress(0.333, desc="Downloading model, typically ~10mins...")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/download-model.sh", shell=True)
                if rc != 0:
                    gr.Warning("You may have improper configurations set for this mode. Check the Output in the AI Workbench UI for details.")
                    msg = "Generate Model Repo"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        model_repo_generate: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                progress(0.667, desc="Generating Model Repo, typically ~5mins...")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/model-repo-generator.sh", shell=True)
                if rc == 0:
                    msg = "Model Repo Generated"
                    colors = "primary"
                    interactive = False
                    start_interactive = True if (start == "Start Microservice") else False
                    stop_interactive = True if (stop == "Stop Microservice") else False
                else: 
                    gr.Warning("Ran into an error generating the model repo. Check the Output in the AI Workbench UI for details.")
                    msg = "Generate Model Repo"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
            progress(0.8, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                model_repo_generate: gr.update(value=msg, variant=colors, interactive=interactive),
                start_local_nim: gr.update(interactive=start_interactive),
                stop_local_nim: gr.update(interactive=stop_interactive),
            }
        
        model_repo_generate.click(_toggle_model_repo_generate,
                             [model_repo_generate, nim_local_model_id, start_local_nim, stop_local_nim], 
                             [model_repo_generate, start_local_nim, stop_local_nim, msg])
        
        def _toggle_local_nim(btn: str, model: str, model_repo_gen: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Start Microservice":
                progress(0.2, desc="Initializing Task")
                time.sleep(0.5)
                progress(0.4, desc="Checking user configs...")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/preflight.sh", shell=True)
                if rc != 0:
                    gr.Warning("You may have improper configurations set for this mode. Check the Output in the AI Workbench UI for details.")
                    out = ["Internal Server Error, Try Again", "Stop Microservice"]
                    colors = ["stop", "secondary"]
                    interactive = [False, True, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value=""
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    model_repo_gen_interactive = True
                    return {
                        start_local_nim: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                        stop_local_nim: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                        nim_local_model_id: gr.update(interactive=interactive[2]),
                        remote_nim_msg: gr.update(value=value),
                        which_nim_tab: submittable, 
                        submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                        model_repo_generate: gr.update(interactive=model_repo_gen_interactive),
                        msg: gr.update(interactive=True, 
                                       placeholder=("Enter text and press SUBMIT" if interactive[3] else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode.")),
                    }
                progress(0.6, desc="Starting Microservice, may take a moment")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/start-local-nim.sh " + model, shell=True)
                if rc == 0:
                    out = ["Microservice Started", "Stop Microservice"]
                    colors = ["primary", "secondary"]
                    interactive = [False, True, False, True]
                    model_ip = ["local_nim"]
                    model_id = [model]
                    value="<br />Stop the local microservice before using a remote microservice."
                    submit_value = "Submit"
                    submittable = 0
                    model_repo_gen_interactive = False
                else: 
                    gr.Warning("Ran into an error starting up the NIM Container. Double check the model name spelling. See Troubleshooting for details. ")
                    out = ["Internal Server Error, Try Again", "Stop Microservice"]
                    colors = ["stop", "secondary"]
                    interactive = [False, True, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value=""
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    model_repo_gen_interactive = True if model_repo_gen == "Generate Model Repo" else False
                progress(0.8, desc="Cleaning Up")
                time.sleep(0.5)
            elif btn == "Stop Microservice":
                progress(0.25, desc="Initializing")
                time.sleep(0.5)
                progress(0.5, desc="Stopping Microservice")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/stop-local-nim.sh ", shell=True)
                if rc == 0:
                    out = ["Start Microservice", "Microservice Stopped"]
                    colors = ["secondary", "primary"]
                    interactive = [True, False, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value="<br />Enter the details below. Then start chatting!"
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    model_repo_gen_interactive = True if model_repo_gen == "Generate Model Repo" else False
                else: 
                    out = ["Start Microservice", "Internal Server Error, Try Again"]
                    colors = ["secondary", "stop"]
                    interactive = [True, False, True, True]
                    model_ip = ["local_nim"]
                    model_id = [model]
                    value=""
                    submit_value = "Submit"
                    submittable = 0
                    model_repo_gen_interactive = False
                progress(0.75, desc="Cleaning Up")
                time.sleep(0.5)
            return {
                start_local_nim: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                stop_local_nim: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                nim_local_model_id: gr.update(interactive=interactive[2]),
                remote_nim_msg: gr.update(value=value),
                which_nim_tab: submittable, 
                submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                model_repo_generate: gr.update(interactive=model_repo_gen_interactive),
                msg: gr.update(interactive=True, 
                               placeholder=("Enter text and press SUBMIT" if interactive[3] else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode.")),
            }

        start_local_nim.click(_toggle_local_nim, 
                                 [start_local_nim, 
                                  nim_local_model_id,
                                  model_repo_generate], 
                                 [start_local_nim, 
                                  stop_local_nim, 
                                  nim_local_model_id, 
                                  remote_nim_msg,
                                  which_nim_tab, 
                                  submit_btn,
                                  model_repo_generate,
                                  msg])
        stop_local_nim.click(_toggle_local_nim, 
                                 [stop_local_nim, 
                                  nim_local_model_id,
                                  model_repo_generate], 
                                 [start_local_nim, 
                                  stop_local_nim, 
                                  nim_local_model_id, 
                                  remote_nim_msg,
                                  which_nim_tab, 
                                  submit_btn,
                                  model_repo_generate,
                                  msg])

        def lock_tabs(btn: str, start_local_server: str, which_nim_tab: int, nvcf_model_family: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Local System":
                if start_local_server == "Server Started":
                    interactive=True
                else: 
                    interactive=False
                return {
                    tabs: gr.update(selected=0),
                    msg: gr.update(interactive=True, 
                                   placeholder=("Enter text and press SUBMIT" if interactive else "[NOT READY] Start the Local Inference Server OR Select a Different Inference Mode.")),
                    inference_mode: gr.update(info="To use your LOCAL GPU for inference, start the Local Inference Server before making a query."),
                    submit_btn: gr.update(value="Submit" if interactive else "[NOT READY] Submit", interactive=interactive),
                }
            elif btn == "Cloud Endpoint":
                if nvcf_model_family == "Select":
                    interactive=False
                else: 
                    interactive=True
                return {
                    tabs: gr.update(selected=1),
                    msg: gr.update(interactive=True, placeholder=("Enter text and press SUBMIT" if interactive else "[NOT READY] Select a model OR Select a Different Inference Mode.")),
                    inference_mode: gr.update(info="To use a CLOUD endpoint for inference, select the desired model before making a query."),
                    submit_btn: gr.update(value="Submit" if interactive else "[NOT READY] Submit", interactive=interactive),
                }
            elif btn == "Self-Hosted Microservice":
                return {
                    tabs: gr.update(selected=2),
                    msg: gr.update(interactive=True, placeholder="Enter text and press SUBMIT" if (which_nim_tab == 0) else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."),
                    inference_mode: gr.update(info="To use a MICROSERVICE for inference, input the endpoint (and/or model) before making a query."),
                    submit_btn: gr.update(value="Submit" if (which_nim_tab == 0) else "[NOT READY] Submit",
                                          interactive=True if (which_nim_tab == 0) else False),
                }
        
        inference_mode.change(lock_tabs, [inference_mode, start_local_server, which_nim_tab, nvcf_model_family], [tabs, msg, inference_mode, submit_btn])

        def _toggle_kb(btn: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Clear Database":
                progress(0.25, desc="Initializing Task")
                time.sleep(0.25)
                progress(0.5, desc="Clearing Vector Database")
                success = clear_knowledge_base()
                if success:
                    out = ["Clear Database"]
                    colors = ["secondary"]
                    interactive = [True]
                    progress(0.75, desc="Success!")
                    time.sleep(0.75)
                else: 
                    gr.Warning("Your files may still be present in the database. Try again.")
                    out = ["Error Clearing Vector Database"]
                    colors = ["stop"]
                    interactive = [True]
                    progress(0.75, desc="Error, try again.")
                    time.sleep(0.75)
            else: 
                out = ["Clear Database"]
                colors = ["secondary"]
                interactive = [True]
            return {
                file_output: gr.update(value=None, 
                                       interactive=True, 
                                       show_label=False, 
                                       file_types=["text",
                                                   ".pdf",
                                                   ".html",
                                                   ".doc",
                                                   ".docx",
                                                   ".txt",
                                                   ".odt",
                                                   ".rtf",
                                                   ".tex"], 
                                       file_count="multiple"),
                clear_docs: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                kb_checkbox: gr.update(value=None),
            }
            
        clear_docs.click(_toggle_kb, [clear_docs], [clear_docs, file_output, kb_checkbox, msg])

        def vdb_select(inf_mode: str, start_local: str, vdb_active: bool, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Awaiting Vector DB Readiness")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                if not vdb_active:
                    gr.Info("The Vector Database is now ready for file upload. ")
                interactive=True
            else: 
                gr.Warning("The Vector Database has timed out. Check Output > Chat on AI Workbench for the full logs. ")
                interactive=False
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            return [True if rc == 0 else False,
                    gr.update(interactive=interactive), 
                    gr.update(interactive=interactive)]
            
        vdb_settings.select(vdb_select, [inference_mode, start_local_server, vdb_active], [vdb_active, file_output, clear_docs])

        def document_upload(files, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Polling Vector DB Status")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                progress(0.75, desc="Pushing uploaded files to DB...")
                file_paths = upload_file(files, client)
                success=True
            else: 
                gr.Warning("Hang Tight! The Vector DB may be temporarily busy. Give it a moment, and then try again. ")
                file_paths = None
                success=False
            return {
                file_output: gr.update(value=file_paths), 
                kb_checkbox: gr.update(value="Toggle to use Vector Database" if success else None),
            }

        file_output.upload(document_upload, file_output, [file_output, kb_checkbox])

        def toggle_rag_start(btn: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
            rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
            if rc == 2:
                gr.Info("Inferencing is ready, but the Vector DB may still be spinning up. This can take a few moments to complete. ")
                visibility = [False, True, True]
                interactive = [False, True, True, False]
                submit_value="[NOT READY] Submit"
            elif rc == 0:
                visibility = [False, True, True]
                interactive = [False, True, True, False]
                submit_value="[NOT READY] Submit"
            else:
                gr.Warning("Something went wrong. Check the Output in AI Workbench, or try again. ")
                visibility = [True, True, True]
                interactive = [False, False, False, False]
                submit_value="[NOT READY] Submit"
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            return {
                setup_settings: gr.update(visible=visibility[0], interactive=interactive[0]), 
                inf_settings: gr.update(visible=visibility[1], interactive=interactive[1]),
                vdb_settings: gr.update(visible=visibility[2], interactive=interactive[2]),
                submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                msg: gr.update(interactive=True, placeholder="[NOT READY] Select a model OR Select a Different Inference Mode." if rc != 1 else "Enter text and press SUBMIT"),
            }
        
        rag_start_button.click(toggle_rag_start, [rag_start_button], [setup_settings, inf_settings, vdb_settings, submit_btn, msg])

        def toggle_remote_ms() -> Dict[gr.component, Dict[Any, Any]]:
            return {
                which_nim_tab: 0, 
                is_local_nim: False, 
                submit_btn: gr.update(value="Submit", interactive=True),
                msg: gr.update(placeholder="Enter text and press SUBMIT")
            }
        
        remote_microservice.select(toggle_remote_ms, None, [which_nim_tab, is_local_nim, submit_btn, msg])

        def toggle_local_ms(start_btn: str, stop_btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if (start_btn == "Microservice Started"):
                interactive = True
                submit_value = "Submit"
                msg_value = "Enter text and press SUBMIT"
                submittable = 0
            elif (start_btn == "Start Microservice" and stop_btn == "Stop Microservice"): 
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."
                submittable = 1
            else:
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."
                submittable = 1
            return {
                which_nim_tab: submittable, 
                is_local_nim: True, 
                submit_btn: gr.update(value=submit_value, interactive=interactive),
                msg: gr.update(placeholder=msg_value)
            }
        
        local_microservice.select(toggle_local_ms, [start_local_nim, stop_local_nim], [which_nim_tab, is_local_nim, submit_btn, msg])
        
        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [kb_checkbox, 
                               inference_mode, 
                               nvcf_model_id, 
                               nim_model_ip, 
                               nim_model_port, 
                               nim_local_model_id,
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               start_local_server,
                               local_model_id,
                               msg, 
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history]
        )
        submit_btn.click(
            _my_build_stream, [kb_checkbox, 
                               inference_mode, 
                               nvcf_model_id, 
                               nim_model_ip, 
                               nim_model_port, 
                               nim_local_model_id,
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               start_local_server,
                               local_model_id,
                               msg, 
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history]
        )

    page.queue()
    return page

def _stream_predict(
    client: chat_client.ChatClient,
    use_knowledge_base: List[str],
    inference_mode: str,
    nvcf_model_id: str,
    nim_model_ip: str,
    nim_model_port: str,
    nim_local_model_id: str, 
    nim_model_id: str,
    is_local_nim: bool,
    num_token_slider: float, 
    temp_slider: float, 
    start_local_server: str,
    local_model_id: str,
    question: str,
    metrics_history: dict,
    chat_history: List[Tuple[str, str]],
) -> Any:
    """Make a prediction of the response to the prompt."""
    chunks = ""
    _LOGGER.info(
        "processing inference request - %s",
        str({"prompt": question, "use_knowledge_base": False if len(use_knowledge_base) == 0 else True}),
    )

    if (inference_to_config(inference_mode) == "microservice" and
        (len(nim_model_ip) == 0 or len(nim_model_id) == 0) and 
        is_local_nim == False):
        yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nVerify your settings are nonempty and correct before submitting a query. "]], None, gr.update(value=metrics_history), metrics_history
    else:
        try:
            documents: Union[None, List[Dict[str, Union[str, float]]]] = None
            response_num = len(metrics_history.keys())
            retrieval_ftime = ""
            e2e_stime = time.time()
            if len(use_knowledge_base) != 0:
                retrieval_stime = time.time()
                documents = client.search(question)
                retrieval_ftime = str((time.time() - retrieval_stime) * 1000).split('.', 1)[0]
        
            chunk_num = 0
            for chunk in client.predict(question, 
                                        inference_to_config(inference_mode), 
                                        local_model_id,
                                        cloud_to_config(nvcf_model_id), 
                                        "local_nim" if is_local_nim else nim_model_ip, 
                                        "9999" if is_local_nim else nim_model_port, 
                                        nim_local_model_id if is_local_nim else nim_model_id,
                                        temp_slider,
                                        False if len(use_knowledge_base) == 0 else True, 
                                        int(num_token_slider)):
                if chunk_num == 0:
                    chunk_num += 1
                    ttft = chunk
                    updated_metrics_history = metrics_history.update({str(response_num): {"inference_mode": inference_to_config(inference_mode),
                                                                                                         "model": nvcf_model_id if inference_to_config(inference_mode)=="cloud" else (local_model_id if inference_to_config(inference_mode)=="local" else (nim_local_model_id if inference_to_config(inference_mode) and is_local_nim else nim_model_id)),
                                                                                                         "Retrieval time": "N/A" if len(retrieval_ftime) == 0 else retrieval_ftime + "ms",
                                                                                                         "Time to First Token (TTFT)": ttft + "ms",
                                                                                                        }})
                    yield "", chat_history, documents, gr.update(value=updated_metrics_history), updated_metrics_history
                else:
                    chunks += chunk
                    chunk_num += 1
                yield "", chat_history + [[question, chunks]], documents, gr.update(value=metrics_history), metrics_history
            e2e_ftime = str((time.time() - e2e_stime) * 1000).split('.', 1)[0]
            gen_time = int(e2e_ftime) - int(ttft) if len(retrieval_ftime) == 0 else int(e2e_ftime) - int(ttft) - int(retrieval_ftime)
            metrics_history.get(str(response_num)).update({"Generation Time": str(gen_time) + "ms", 
                                                           "End to End Time (E2E)": e2e_ftime + "ms", 
                                                           "Generated Tokens": str(chunk_num - 1) + " tokens", 
                                                           "Generated Tokens Per Second": str(round(chunk_num / (gen_time / 1000), 1)) + " tokens/sec"})
            yield "", gr.update(show_label=metrics_history), documents, gr.update(value=metrics_history), metrics_history
        except: 
            yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nCheck Output > Chat on the AI Workbench application for full logs. "]], None, gr.update(value=metrics_history), metrics_history
