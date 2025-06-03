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

### This module contains the chatui gui for having a conversation. ###

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
# import torch
import tiktoken
import fnmatch
import traceback

from chatui import assets, chat_client
from chatui.pages import info
from chatui.pages import utils

_LOGGER = logging.getLogger(__name__)
PATH = "/"
TITLE = "Hybrid RAG: Chat UI"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

### Load in CSS here for components that need custom styling. ###

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

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        # create the page header
        gr.Markdown(f"# {TITLE}")

        # Keep track of state we want to persist across user actions
        which_nim_tab = gr.State(0)
        is_local_nim = gr.State(False)
        vdb_active = gr.State(False)
        metrics_history = gr.State({})
        docs_history = gr.State({})

        # Build the Chat Application
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=15, min_width=350):

                # Main chatbot panel. Context and Metrics are hidden until toggled
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=350):
                        chatbot = gr.Chatbot(show_label=False)
                        
                    context = gr.JSON(
                        scale=1,
                        label="Retrieved Context",
                        visible=False,
                        elem_id="contextbox",
                    )
                    
                    metrics = gr.JSON(
                        scale=1,
                        label="Metrics",
                        visible=False,
                        elem_id="contextbox",
                    )

                    docs = gr.JSON(
                        scale=1,
                        label="Documents",
                        visible=False,
                        elem_id="contextbox",
                    )

                # Render the output sliders to customize the generation output. 
                with gr.Tabs(selected=0, visible=False) as out_tabs:
                    with gr.TabItem("Max Tokens in Response", id=0) as max_tokens_in_response:
                        num_token_slider = gr.Slider(0, utils.preset_max_tokens()[1], value=utils.preset_max_tokens()[0], 
                                                     label=info.num_token_label, 
                                                     interactive=True)
                        
                    with gr.TabItem("Temperature", id=1) as temperature:
                        temp_slider = gr.Slider(0, 1, value=0.7, 
                                                label=info.temp_label, 
                                                interactive=True)
                        
                    with gr.TabItem("Top P", id=2) as top_p:
                        top_p_slider = gr.Slider(0.001, 0.999, value=0.999, 
                                                 label=info.top_p_label, 
                                                 interactive=True)
                        
                    with gr.TabItem("Frequency Penalty", id=3) as freq_pen:
                        freq_pen_slider = gr.Slider(-2, 2, value=0, 
                                                    label=info.freq_pen_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Presence Penalty", id=4) as pres_pen:
                        pres_pen_slider = gr.Slider(-2, 2, value=0, 
                                                    label=info.pres_pen_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Hide Output Tools", id=5) as hide_out_tools:
                        gr.Markdown("")

                # Hidden button to expand output sliders, if hidden
                out_tabs_show = gr.Button(value="Show Output Tools", size="sm", visible=True)

                # Render the user input textbox and checkbox to toggle vanilla inference and RAG.
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

                # Render the row of buttons: submit query, clear history, show metrics and contexts
                with gr.Row():
                    submit_btn = gr.Button(value="[NOT READY] Submit", interactive=False)
                    _ = gr.ClearButton([msg, chatbot, metrics, metrics_history], value="Clear history")
                    mtx_show = gr.Button(value="Show Metrics")
                    mtx_hide = gr.Button(value="Hide Metrics", visible=False)
                    ctx_show = gr.Button(value="Show Context")
                    ctx_hide = gr.Button(value="Hide Context", visible=False)

            # Right Column will display the inference and database settings
            with gr.Column(scale=10, min_width=450, visible=True) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:

                    # First tab item is a button to start the RAG backend and unlock other settings
                    with gr.TabItem("Initial Setup", id=0, interactive=False, visible=True) as setup_settings:
                        gr.Markdown("<br> ")
                        gr.Markdown(info.setup)
                        rag_start_button = gr.Button(value="Set Up RAG Backend", variant="primary")
                        gr.Markdown("<br> ")

                    # Second tab item consists of all the inference mode settings
                    with gr.TabItem("Inference Settings", id=1, interactive=False, visible=True) as inf_settings:
                        inference_mode = gr.Radio(["Local System", "Cloud Endpoint", "Self-Hosted Microservice"], 
                                                  label="Inference Mode", 
                                                  info=info.inf_mode_info, 
                                                  value="Cloud Endpoint")
                        
                        # Depending on the selected inference mode, different settings need to get exposed to the user.
                        with gr.Tabs(selected=1) as tabs:

                            # Inference settings for local TGI inference server
                            with gr.TabItem("Local System", id=0, interactive=False, visible=False) as local:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(info.local_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(info.local_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(info.local_trouble)

                                gate_checkbox = gr.CheckboxGroup(
                                    ["Ungated Models", "Gated Models"], 
                                    value=["Ungated Models"], 
                                    label="Select which models types to show", 
                                    interactive = True,
                                    elem_id="rag-inputs")
                                
                                local_model_id = gr.Dropdown(choices = ["nvidia/Llama3-ChatQA-1.5-8B",
                                                                        "microsoft/Phi-3-mini-128k-instruct"], 
                                                             value = "nvidia/Llama3-ChatQA-1.5-8B",
                                                             interactive = True,
                                                             label = "Select a model (or input your own).", 
                                                             allow_custom_value = True, 
                                                             elem_id="rag-inputs")
                                local_model_quantize = gr.Dropdown(choices = ["None",
                                                                              "8-Bit",
                                                                              "4-Bit"], 
                                                                   value = utils.preset_quantization(),
                                                                   interactive = True,
                                                                   label = "Select model quantization.", 
                                                                   elem_id="rag-inputs")
                                
                                with gr.Row(equal_height=True):
                                    download_model = gr.Button(value="Load Model", size="sm")
                                    start_local_server = gr.Button(value="Start Server", interactive=False, size="sm")
                                    stop_local_server = gr.Button(value="Stop Server", interactive=False, size="sm")

                            # Inference settings for cloud endpoints inference mode
                            with gr.TabItem("Cloud Endpoint", id=1, interactive=False, visible=False) as cloud:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(info.cloud_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(info.cloud_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(info.cloud_trouble)
                                
                                nvcf_model_family = gr.Dropdown(choices = ["Select", 
                                                                           "NVIDIA", 
                                                                           "MistralAI", 
                                                                           "Meta", 
                                                                           "Google",
                                                                           "Microsoft", 
                                                                           "Snowflake",
                                                                           "IBM",
                                                                           "Upstage",
                                                                           "AI21 Labs"], 
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

                            # Inference settings for self-hosted microservice inference mode
                            with gr.TabItem("Self-Hosted Microservice", id=2, interactive=False, visible=False) as microservice:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(info.nim_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(info.nim_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(info.nim_trouble)
        
                                remote_nim_msg = gr.Markdown("<br />Enter the details below. Then start chatting!")
                                
                                with gr.Row(equal_height=True):
                                    nim_model_ip = gr.Textbox(value = "hybrid-rag-local-nim-1", 
                                               label = "Microservice Host", 
                                               info = "IP Address running the microservice", 
                                               elem_id="rag-inputs", scale=2)
                                    nim_model_port = gr.Textbox(placeholder = "8000", 
                                               label = "Port", 
                                               info = "Optional, (default: 8000)", 
                                               elem_id="rag-inputs", scale=1)
                                
                                nim_model_id = gr.Textbox(placeholder = "meta/llama-3.1-8b-instruct", 
                                           label = "Model running in microservice.", 
                                           info = "If none specified, defaults to: meta/llama-3.1-8b-instruct", 
                                           elem_id="rag-inputs")

                    # Third tab item consists of database and document upload settings
                    with gr.TabItem("Upload Documents Here", id=2, interactive=False, visible=True) as vdb_settings:
                        
                        gr.Markdown(info.update_kb_info)
                        
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
                            doc_show = gr.Button(value="Show Documents", size="sm")
                            doc_hide = gr.Button(value="Hide Documents", visible=False, size="sm")
                            clear_docs = gr.Button(value="Clear Database", interactive=True, size="sm") 

                    # Final tab item consists of option to collapse the settings to reduce clutter on the UI
                    with gr.TabItem("Hide All Settings", id=3, visible=False) as hide_all_settings:
                        gr.Markdown("")

            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")

        def _toggle_gated(models: List[str]) -> Dict[gr.component, Dict[Any, Any]]:
            """" Event listener to toggle local models displayed to the user. """
            if len(models) == 0:
                choices = []
                selected = ""
            elif len(models) == 1 and models[0] == "Ungated Models":
                choices = ["nvidia/Llama3-ChatQA-1.5-8B",
                           "microsoft/Phi-3-mini-128k-instruct"]
                selected = "nvidia/Llama3-ChatQA-1.5-8B"
            elif len(models) == 1 and models[0] == "Gated Models":
                choices = ["mistralai/Mistral-7B-Instruct-v0.1",
                           "mistralai/Mistral-7B-Instruct-v0.2",
                           "meta-llama/Llama-2-7b-chat-hf",
                           "meta-llama/Meta-Llama-3-8B-Instruct"]
                selected = "mistralai/Mistral-7B-Instruct-v0.1"
            else: 
                choices = ["nvidia/Llama3-ChatQA-1.5-8B", 
                           "microsoft/Phi-3-mini-128k-instruct",
                           "mistralai/Mistral-7B-Instruct-v0.1",
                           "mistralai/Mistral-7B-Instruct-v0.2",
                           "meta-llama/Llama-2-7b-chat-hf",
                           "meta-llama/Meta-Llama-3-8B-Instruct"]
                selected = "nvidia/Llama3-ChatQA-1.5-8B"
            return {
                local_model_id: gr.update(choices=choices, value=selected),
            }

        gate_checkbox.change(_toggle_gated, [gate_checkbox], [local_model_id])
                
        def _toggle_info(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            """" Event listener to toggle context and/or metrics panes visible to the user. """
            if btn == "Show Context":
                out = [True, False, False, False, True, True, False, True, False]
            elif btn == "Hide Context":
                out = [False, False, False, True, False, True, False, True, False]
            elif btn == "Show Metrics":
                out = [False, True, False, True, False, False, True, True, False]
            elif btn == "Hide Metrics":
                out = [False, False, False, True, False, True, False, True, False]
            elif btn == "Show Documents":
                out = [False, False, True, True, False, True, False, False, True]
            elif btn == "Hide Documents":
                out = [False, False, False, True, False, True, False, True, False]
            return {
                context: gr.update(visible=out[0]),
                metrics: gr.update(visible=out[1]),
                docs: gr.update(visible=out[2]),
                ctx_show: gr.update(visible=out[3]),
                ctx_hide: gr.update(visible=out[4]),
                mtx_show: gr.update(visible=out[5]),
                mtx_hide: gr.update(visible=out[6]),
                doc_show: gr.update(visible=out[7]),
                doc_hide: gr.update(visible=out[8]),
            }

        ctx_show.click(_toggle_info, [ctx_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        ctx_hide.click(_toggle_info, [ctx_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        mtx_show.click(_toggle_info, [mtx_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        mtx_hide.click(_toggle_info, [mtx_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        doc_show.click(_toggle_info, [doc_show], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])
        doc_hide.click(_toggle_info, [doc_hide], [context, metrics, docs, ctx_show, ctx_hide, mtx_show, mtx_hide, doc_show, doc_hide])

        def _toggle_hide_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to hide output toolbar from the user. """
            return {
                out_tabs: gr.update(visible=False, selected=0),
                out_tabs_show: gr.update(visible=True),
            }

        hide_out_tools.select(_toggle_hide_out_tools, None, [out_tabs, out_tabs_show])

        def _toggle_show_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to expand output toolbar for the user. """
            return {
                out_tabs: gr.update(visible=True),
                out_tabs_show: gr.update(visible=False),
            }

        out_tabs_show.click(_toggle_show_out_tools, None, [out_tabs, out_tabs_show])

        def _toggle_hide_all_settings() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to hide inference settings pane from the user. """
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])

        def _toggle_show_all_settings() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to expand inference settings pane for the user. """
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=1),
                hidden_settings_column: gr.update(visible=False),
            }

        show_settings.click(_toggle_show_all_settings, None, [settings_column, settings_tabs, hidden_settings_column])

        def _toggle_model_download(btn: str, model: str, start: str, stop: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to download model weights locally for Hugging Face TGI local inference. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
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
            """ Event listener to select different models to use for Hugging Face TGI local inference. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
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

        def _toggle_nvcf_family(family: str) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select a different family of model for cloud endpoint inference. """
            interactive = True
            submit_value = "Submit"
            msg_value = "Enter text and press SUBMIT"
            if family == "NVIDIA":
                choices = ["Llama3 ChatQA-1.5 8B", 
                           "Llama3 ChatQA-1.5 70B", 
                           "Nemotron Mini 4B", 
                           "Nemotron-4 340B Instruct", 
                           "Mistral-NeMo 12B Instruct"]
                value = "Llama3 ChatQA-1.5 8B"
                visible = True
            elif family == "MistralAI":
                choices = ["Mistral 7B Instruct v0.2", 
                           "Mistral 7B Instruct v0.3", 
                           "Mistral Large", 
                           "Mixtral 8x7B Instruct v0.1", 
                           "Mixtral 8x22B Instruct v0.1",
                           "Mistral-NeMo 12B Instruct",
                           "Mamba Codestral 7B v0.1",
                           "Codestral 22B Instruct v0.1"]
                value = "Mistral 7B Instruct v0.2"
                visible = True
            elif family == "Meta":
                choices = ["Llama 2 70B", 
                           "Llama 3 8B", 
                           "Llama 3 70B", 
                           "Llama 3.1 8B", 
                           "Llama 3.1 70B", 
                           "Llama 3.1 405B"]
                value = "Llama 2 70B"
                visible = True
            elif family == "Google":
                choices = ["Gemma 2B", "Gemma 7B", "Code Gemma 7B"]
                value = "Gemma 2B"
                visible = True
            elif family == "Microsoft":
                choices = ["Phi-3 Mini (4k)", 
                           "Phi-3 Mini (128k)", 
                           "Phi-3 Small (8k)", 
                           "Phi-3 Small (128k)", 
                           "Phi-3 Medium (4k)", 
                           "Phi-3 Medium (128k)", 
                           "Phi-3.5 Mini Instruct", 
                           "Phi-3.5 MoE Instruct"]
                value = "Phi-3 Mini (4k)"
                visible = True
            elif family == "Snowflake":
                choices = ["Arctic"]
                value = "Arctic"
                visible = True
            elif family == "IBM":
                choices = ["Granite 8B Code", "Granite 34B Code"]
                value = "Granite 8B Code"
                visible = True
            elif family == "Upstage":
                choices = ["Solar 10.7B Instruct"]
                value = "Solar 10.7B Instruct"
                visible = True
            elif family == "AI21 Labs":
                choices = ["Jamba-1.5 Mini Instruct", "Jamba-1.5 Large Instruct"]
                value = "Jamba-1.5 Mini Instruct"
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
        
        nvcf_model_family.change(_toggle_nvcf_family,
                              [nvcf_model_family], 
                              [nvcf_model_id, submit_btn, msg])

        def _toggle_local_server(btn: str, model: str, quantize: str, download: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to run and/or shut down the Hugging Face TGI local inference server. """
            if model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and btn != "Stop Server" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
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
                                              + model + " " + utils.quant_to_config(quantize), shell=True)
                    if rc == 0:
                        out = ["Server Started", "Stop Server"]
                        colors = ["primary", "secondary"]
                        interactive = [False, True, True, False]
                    else: 
                        gr.Warning("ERR: You may have timed out or are facing memory issues. In AI Workbench, check Output > Chat for details.")
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

        def _lock_tabs(btn: str, 
                       start_local_server: str, 
                       which_nim_tab: int, 
                       nvcf_model_family: str, 
                       progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to lock settings options with the user selected inference mode. """
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
        
        inference_mode.change(_lock_tabs, [inference_mode, start_local_server, which_nim_tab, nvcf_model_family], [tabs, msg, inference_mode, submit_btn])

        def _toggle_kb(btn: str, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to clear the vector database of all documents. """
            if btn == "Clear Database":
                progress(0.25, desc="Initializing Task")
                update_docs_uploaded = docs_uploaded
                time.sleep(0.25)
                progress(0.5, desc="Clearing Vector Database")
                success = utils.clear_knowledge_base()
                if success:
                    out = ["Clear Database"]
                    colors = ["secondary"]
                    interactive = [True]
                    progress(0.75, desc="Success!")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Deleted"})
                    time.sleep(0.5)
                else: 
                    gr.Warning("Your files may still be present in the database. Try again.")
                    out = ["Error Clearing Vector Database"]
                    colors = ["stop"]
                    interactive = [True]
                    progress(0.75, desc="Error, try again.")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Unknown"})
                    time.sleep(0.5)
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
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
            }
            
        clear_docs.click(_toggle_kb, [clear_docs, docs_history], [clear_docs, file_output, kb_checkbox, msg, docs, docs_history])

        def _vdb_select(inf_mode: str, start_local: str, vdb_active: bool, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select the vector database settings top-level tab. """
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
            
        vdb_settings.select(_vdb_select, [inference_mode, start_local_server, vdb_active], [vdb_active, file_output, clear_docs])

        def _document_upload(files, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to upload documents to the vector database. """
            progress(0.25, desc="Initializing Task")
            update_docs_uploaded = docs_uploaded
            time.sleep(0.25)
            progress(0.5, desc="Polling Vector DB Status")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                progress(0.75, desc="Pushing uploaded files to DB...")
                file_paths = utils.upload_file(files, client)
                success=True
                for file in file_paths:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Uploaded Successfully"})
            else: 
                gr.Warning("Hang Tight! The Vector DB may be temporarily busy. Give it a moment, and then try again. ")
                file_paths = None
                success=False
                file_names = [file.name for file in files]
                for file in file_names:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Failed to Upload"})
            return {
                file_output: gr.update(value=file_paths), 
                kb_checkbox: gr.update(value="Toggle to use Vector Database" if success else None),
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
            }

        file_output.upload(_document_upload, [file_output, docs_history], [file_output, kb_checkbox, docs, docs_history])

        def _toggle_rag_start(btn: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to initialize the RAG backend API server and start warming up the vector database. """
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
            rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
            if rc == 2:
                gr.Info("Inferencing is ready, but the Vector DB may still be spinning up. This can take a few moments to complete. ")
                visibility = [False, True, True, True]
                interactive = [False, True, True, False]
                submit_value="[NOT READY] Submit"
            elif rc == 0:
                visibility = [False, True, True, True]
                interactive = [False, True, True, False]
                submit_value="[NOT READY] Submit"
            else:
                gr.Warning("Something went wrong. Check the Output in AI Workbench, or try again. ")
                visibility = [True, True, True, False]
                interactive = [False, False, False, False]
                submit_value="[NOT READY] Submit"
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            return {
                setup_settings: gr.update(visible=visibility[0], interactive=interactive[0]), 
                inf_settings: gr.update(visible=visibility[1], interactive=interactive[1]),
                vdb_settings: gr.update(visible=visibility[2], interactive=interactive[2]),
                submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                hide_all_settings: gr.update(visible=visibility[3]),
                msg: gr.update(interactive=True, placeholder="[NOT READY] Select a model OR Select a Different Inference Mode." if rc != 1 else "Enter text and press SUBMIT"),
            }
        
        rag_start_button.click(_toggle_rag_start, [rag_start_button], [setup_settings, inf_settings, vdb_settings, submit_btn, hide_all_settings, msg])

        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [kb_checkbox, 
                               inference_mode, 
                               nvcf_model_id, 
                               nim_model_ip, 
                               nim_model_port, 
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               top_p_slider, 
                               freq_pen_slider, 
                               pres_pen_slider,
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
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               top_p_slider, 
                               freq_pen_slider, 
                               pres_pen_slider,
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
    nim_model_id: str,
    is_local_nim: bool,
    num_token_slider: float, 
    temp_slider: float, 
    top_p_slider: float, 
    freq_pen_slider: float, 
    pres_pen_slider: float, 
    start_local_server: str,
    local_model_id: str,
    question: str,
    metrics_history: dict,
    chat_history: List[Tuple[str, str]],
) -> Any:
    """
    Make a prediction of the response to the prompt.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
        use_knowledge_base (List[str]): Whether or not the vector db should be invoked for this query
        inference_mode (str): The inference mode selected for this query
        nvcf_model_id (str): The cloud endpoint selected for this query
        nim_model_ip (str): The ip address running the remote nim selected for this query
        nim_model_port (str): The port for the remote nim selected for this query
        nim_model_id (str): The model name for remote nim selected for this query
        is_local_nim (bool): Whether to run the query as local or remote nim
        num_token_slider (float): max number of tokens to generate
        temp_slider (float): temperature selected for this query
        top_p_slider (float): top_p selected for this query
        freq_pen_slider (float): frequency penalty selected for this query 
        pres_pen_slider (float): presence penalty selected for this query
        start_local_server (str): local TGI server status
        local_model_id (str): model name selected for local TGI inference of this query
        question (str): user prompt
        metrics_history (dict): current list of generated metrics
        chat_history (List[Tuple[str, str]]): current history of chatbot messages
    
    Returns:
        (Dict[gr.component, Dict[Any, Any]]): Gradio components to update.
    """
    chunks = ""
    _LOGGER.info(
        "processing inference request - %s",
        str({"prompt": question, "use_knowledge_base": False if len(use_knowledge_base) == 0 else True}),
    )

    # Input validation for remote microservice settings
    if (utils.inference_to_config(inference_mode) == "microservice" and
        (len(nim_model_ip) == 0) and 
        is_local_nim == False):
        yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: Hostname/IP field cannot be empty. "]], None, gr.update(value=metrics_history), metrics_history

    # Inputs are validated, can proceed with generating a response to the user query.
    else:

        # Try to send a request for the query
        try:
            documents: Union[None, List[Dict[str, Union[str, float]]]] = None
            response_num = len(metrics_history.keys())
            retrieval_ftime = ""
            chunks = ""
            e2e_stime = time.time()
            if len(use_knowledge_base) != 0:
                retrieval_stime = time.time()
                documents = client.search(question)
                retrieval_ftime = str((time.time() - retrieval_stime) * 1000).split('.', 1)[0]
    
            # Generate the output
            chunk_num = 0
            for chunk in client.predict(question, 
                                        utils.inference_to_config(inference_mode), 
                                        local_model_id,
                                        utils.cloud_to_config(nvcf_model_id), 
                                        nim_model_ip, 
                                        nim_model_port, 
                                        nim_model_id,
                                        temp_slider,
                                        top_p_slider,
                                        freq_pen_slider,
                                        pres_pen_slider,
                                        False if len(use_knowledge_base) == 0 else True, 
                                        int(num_token_slider)):
                
                # The first chunk returned will always be the time to first token. Let's process that first.
                if chunk_num == 0:
                    chunk_num += 1
                    ttft = chunk
                    updated_metrics_history = utils.get_initial_metrics(metrics_history, response_num, inference_mode, nvcf_model_id, local_model_id, 
                                                                        is_local_nim, nim_model_id, retrieval_ftime, ttft)
                    yield "", chat_history, documents, gr.update(value=updated_metrics_history), updated_metrics_history
                
                # Every next chunk will be the generated response. Let's append to the output and render it in real time. 
                else:
                    chunks += chunk
                    chunk_num += 1
                yield "", chat_history + [[question, chunks]], documents, gr.update(value=metrics_history), metrics_history
    
            # With final output generated, run some final calculations and display them as metrics to the user
            gen_time, e2e_ftime, tokens, tokens_sec, itl = utils.get_final_metrics(time.time(), e2e_stime, ttft, retrieval_ftime, chunks)
            metrics_history.get(str(response_num)).update({"Generation Time": gen_time + "ms", 
                                                           "End to End Time (E2E)": e2e_ftime + "ms", 
                                                           "Tokens (est.)": tokens + " tokens", 
                                                           "Tokens/Second (est.)": tokens_sec + " tokens/sec", 
                                                           "Inter-Token Latency (est.)": itl + " ms"})
            yield "", gr.update(show_label=False), documents, gr.update(value=metrics_history), metrics_history

        # Catch any exceptions and direct the user to the logs/output. 
        except Exception as e: 
            yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: " + traceback.format_exc()]], None, gr.update(value=metrics_history), metrics_history
