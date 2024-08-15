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

"""LLM Chains for executing Retrival Augmented Generation."""
import base64
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional

import openai
import torch
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference

import logging
import mimetypes
import typing
import requests
import re

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from llama_index.embeddings import LangchainEmbedding
from llama_index import (
    Prompt,
    ServiceContext,
    VectorStoreIndex,
    download_loader,
    set_global_service_context,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.llms import LangChainLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import StreamingResponse, Response
from llama_index.schema import MetadataMode
from llama_index.utils import globals_helper, get_tokenizer
from llama_index.vector_stores import MilvusVectorStore, SimpleVectorStore

from chain_server import configuration
# from chain_server.trt_llm import TensorRTLLM
from chain_server.nvcf_llm import NvcfLLM

if TYPE_CHECKING:
    from llama_index.indices.base_retriever import BaseRetriever
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore
    from llama_index.types import TokenGen

    from chain_server.configuration_wizard import ConfigWizard

from chain_server import chat_templates

TEXT_SPLITTER_MODEL = "intfloat/e5-large-v2"
TEXT_SPLITTER_CHUNCK_SIZE = 510
TEXT_SPLITTER_CHUNCK_OVERLAP = 200
EMBEDDING_MODEL = "intfloat/e5-large-v2"
DEFAULT_NUM_TOKENS = 50
DEFAULT_MAX_CONTEXT = 800


class LimitRetrievedNodesLength(BaseNodePostprocessor):
    """Llama Index chain filter to limit token lengths."""

    def _postprocess_nodes(
        self, nodes: List["NodeWithScore"] = [], query_bundle: Optional["QueryBundle"] = None
    ) -> List["NodeWithScore"]:
        """Filter function."""
        included_nodes = []
        current_length = 0
        limit = DEFAULT_MAX_CONTEXT
        tokenizer = get_tokenizer()

        for node in nodes:
            current_length += len(
                tokenizer(
                    node.get_content(metadata_mode=MetadataMode.LLM)
                )
            )
            if current_length > limit:
                break
            included_nodes.append(node)

        return included_nodes


@lru_cache
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


@lru_cache
def get_llm(inference_mode: str, nvcf_model_id: str, nim_model_ip: str, num_tokens: int, temp: float, top_p: float, freq_pen: float) -> LangChainLLM:
    """Create the LLM connection."""
    
    if inference_mode == "local":
        inference_server_url_local = "http://127.0.0.1:9090/"

        llm_local = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            max_new_tokens=num_tokens,
            top_k=10,
            top_p=top_p,
            typical_p=0.95,
            temperature=temp,
            repetition_penalty=(freq_pen/8) + 1,   # Reasonable mapping of OpenAI API Spec to HF Spec
            streaming=True
        )
    else: 
        inference_server_url_local = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else "http://" + nim_model_ip + ":8000/v1/"
        
        llm_local = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            max_new_tokens=num_tokens,
            top_k=10,
            top_p=top_p,
            typical_p=0.95,
            temperature=temp,
            repetition_penalty=(freq_pen/8) + 1,   # Reasonable mapping of OpenAI API Spec to HF Spec
            streaming=True
        )

    return LangChainLLM(llm=llm_local)


@lru_cache
def get_embedding_model() -> LangchainEmbedding:
    """Create the embedding model."""
    model_kwargs = {"device": "cpu"}
    device_str = os.environ.get('EMBEDDING_DEVICE', "cuda:1")
    if torch.cuda.is_available():
        model_kwargs["device"] = device_str

    encode_kwargs = {"normalize_embeddings": False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Load in a specific embedding model
    return LangchainEmbedding(hf_embeddings)

@lru_cache
def get_vector_index() -> VectorStoreIndex:
    """Create the vector db index."""
    config = get_config()
    vector_store = MilvusVectorStore(uri=config.milvus, dim=1024, overwrite=False)
    #vector_store = SimpleVectorStore()
    return VectorStoreIndex.from_vector_store(vector_store)


@lru_cache
def get_doc_retriever(num_nodes: int = 4) -> "BaseRetriever":
    """Create the document retriever."""
    index = get_vector_index()
    return index.as_retriever(similarity_top_k=num_nodes)


@lru_cache
def set_service_context(inference_mode: str, nvcf_model_id: str, nim_model_ip: str, num_tokens: int, temp: float, top_p: float, freq_pen: float) -> None:
    """Set the global service context."""
    service_context = ServiceContext.from_defaults(
        llm=get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp, top_p, freq_pen), embed_model=get_embedding_model()
    )
    set_global_service_context(service_context)

def add_http_prefix(input_string: str) -> str:
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    if not input_string.startswith(('http://', 'https://')):
        if re.match(ip_pattern, input_string):  # String is an IPv4 Address; assume http
            input_string = 'http://' + input_string
        elif input_string == "local_nim":       # String is a local NIM; assume http
            input_string = 'http://' + input_string
        else:                                   # String is a general alphanumeric hostname; assume https
            input_string = 'https://' + input_string
    if input_string.endswith('/'): 
        input_string = input_string[:-1]
    return input_string

def llm_chain_streaming(
    context: str, 
    question: str, 
    num_tokens: int, 
    inference_mode: str, 
    local_model_id: str,
    nvcf_model_id: str, 
    nim_model_ip: str,
    nim_model_port: str, 
    nim_model_id: str,
    temp: float,
    top_p: float,
    freq_pen: float,
    pres_pen: float,
) -> Generator[str, None, None]:
    """Execute a simple LLM chain using the components defined above."""
    set_service_context(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp, top_p, freq_pen)

    if inference_mode == "local":
        if "nvidia" in local_model_id:
            prompt = chat_templates.NVIDIA_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif "Llama-3" in local_model_id:
            prompt = chat_templates.LLAMA_3_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif "Llama-2" in local_model_id:
            prompt = chat_templates.LLAMA_2_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif "microsoft" in local_model_id:
            prompt = chat_templates.MICROSOFT_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif "mistralai" in local_model_id:
            prompt = chat_templates.MISTRAL_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        else: 
            prompt = chat_templates.NVIDIA_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        start = time.time()
        response = get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp, top_p, freq_pen).stream_complete(prompt, max_new_tokens=num_tokens)
        perf = time.time() - start
        yield str(perf * 1000).split('.', 1)[0]
        gen_response = (resp.delta for resp in response)
        for chunk in gen_response:
            if "<|eot_id|>" not in chunk:
                yield chunk
            else:
                break
    else:
        if inference_mode == "cloud" and "llama3" in nvcf_model_id:
            prompt = chat_templates.LLAMA_3_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif inference_mode == "cloud" and "llama2" in nvcf_model_id:
            prompt = chat_templates.LLAMA_2_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif inference_mode == "cloud" and "mistral" in nvcf_model_id:
            prompt = chat_templates.MISTRAL_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        elif inference_mode == "cloud" and "microsoft" in nvcf_model_id:
            prompt = chat_templates.MICROSOFT_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        else:
            prompt = chat_templates.GENERIC_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        openai.api_key = os.environ.get('NVCF_RUN_KEY') if inference_mode == "cloud" else "xyz"
        openai.base_url = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else add_http_prefix(nim_model_ip) + ":" + ("8000" if len(nim_model_port) == 0 else nim_model_port) + "/v1/"

        start = time.time()
        completion = openai.chat.completions.create(
          model= nvcf_model_id if inference_mode == "cloud" else ("meta/llama3-8b-instruct" if len(nim_model_id) == 0 else nim_model_id),
          temperature=temp,
          top_p=top_p,
          # frequency_penalty=freq_pen,   # Some models have yet to roll out support for these params
          # presence_penalty=pres_pen,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens, 
          stream=True,
        ) if inference_mode == "cloud" and ("microsoft" in nvcf_model_id or "nemotron" in nvcf_model_id) else openai.chat.completions.create(
          model= nvcf_model_id if inference_mode == "cloud" else ("meta/llama3-8b-instruct" if len(nim_model_id) == 0 else nim_model_id),
          temperature=temp,
          top_p=top_p,
          frequency_penalty=freq_pen,
          presence_penalty=pres_pen,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens, 
          stream=True,
        )
        perf = time.time() - start
        yield str(perf * 1000).split('.', 1)[0]
        
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield str(content)
            else:
                continue

def rag_chain_streaming(prompt: str, 
                        num_tokens: int, 
                        inference_mode: str, 
                        local_model_id: str,
                        nvcf_model_id: str, 
                        nim_model_ip: str,
                        nim_model_port: str, 
                        nim_model_id: str,
                        temp: float,
                        top_p: float,
                        freq_pen: float,
                        pres_pen: float) -> "TokenGen":
    """Execute a Retrieval Augmented Generation chain using the components defined above."""
    set_service_context(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp, top_p, freq_pen)

    if inference_mode == "local":
        get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp, top_p, freq_pen).llm.max_new_tokens = num_tokens  # type: ignore
        nodes = get_doc_retriever(num_nodes=2).retrieve(prompt)
        docs = []
        for node in nodes: 
            docs.append(node.get_text())
        if "nvidia" in local_model_id:
            prompt = chat_templates.NVIDIA_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif "Llama-3" in local_model_id:
            prompt = chat_templates.LLAMA_3_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif "Llama-2" in local_model_id:
            prompt = chat_templates.LLAMA_2_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif "microsoft" in local_model_id:
            prompt = chat_templates.MICROSOFT_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif "mistralai" in local_model_id:
            prompt = chat_templates.MISTRAL_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        else: 
            prompt = chat_templates.NVIDIA_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        start = time.time()
        response = get_llm(inference_mode, 
                           nvcf_model_id, 
                           nim_model_ip, 
                           num_tokens, 
                           temp, 
                           top_p, 
                           freq_pen).stream_complete(prompt, max_new_tokens=num_tokens)
        perf = time.time() - start
        yield str(perf * 1000).split('.', 1)[0]
        gen_response = (resp.delta for resp in response)
        for chunk in gen_response:
            if "<|eot_id|>" not in chunk:
                yield chunk
            else:
                break
    else: 
        openai.api_key = os.environ.get('NVCF_RUN_KEY') if inference_mode == "cloud" else "xyz"
        openai.base_url = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else add_http_prefix(nim_model_ip) + ":" + ("8000" if len(nim_model_port) == 0 else nim_model_port) + "/v1/"
        num_nodes = 1 if ((inference_mode == "cloud" and nvcf_model_id == "playground_llama2_13b") or (inference_mode == "cloud" and nvcf_model_id == "playground_llama2_70b")) else 2
        
        nodes = get_doc_retriever(num_nodes=num_nodes).retrieve(prompt)
        docs = []
        for node in nodes: 
            docs.append(node.get_text())
        if inference_mode == "cloud" and "llama3" in nvcf_model_id:
            prompt = chat_templates.LLAMA_3_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif inference_mode == "cloud" and "llama2" in nvcf_model_id:
            prompt = chat_templates.LLAMA_2_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif inference_mode == "cloud" and "mistral" in nvcf_model_id:
            prompt = chat_templates.MISTRAL_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        elif inference_mode == "cloud" and "microsoft" in nvcf_model_id:
            prompt = chat_templates.MICROSOFT_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        else:
            prompt = chat_templates.GENERIC_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        start = time.time()
        completion = openai.chat.completions.create(
          model= nvcf_model_id if inference_mode == "cloud" else ("meta/llama3-8b-instruct" if len(nim_model_id) == 0 else nim_model_id),
          temperature=temp,
          top_p=top_p,
          # frequency_penalty=freq_pen,   # Some models have yet to roll out support for these params
          # presence_penalty=pres_pen,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens, 
          stream=True,
        ) if inference_mode == "cloud" and ("microsoft" in nvcf_model_id or "nemotron" in nvcf_model_id) else openai.chat.completions.create(
          model=nvcf_model_id if inference_mode == "cloud" else ("meta/llama3-8b-instruct" if len(nim_model_id) == 0 else nim_model_id),
          temperature=temp,
          top_p=top_p,
          frequency_penalty=freq_pen,
          presence_penalty=pres_pen,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens,
          stream=True
        )
        perf = time.time() - start
        yield str(perf * 1000).split('.', 1)[0]
        
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield str(content)
            else:
                continue

def is_base64_encoded(s: str) -> bool:
    """Check if a string is base64 encoded."""
    try:
        # Attempt to decode the string as base64
        decoded_bytes = base64.b64decode(s)
        # Encode the decoded bytes back to a string to check if it's valid
        decoded_str = decoded_bytes.decode("utf-8")
        # If the original string and the decoded string match, it's base64 encoded
        return s == base64.b64encode(decoded_str.encode("utf-8")).decode("utf-8")
    except Exception:  # pylint:disable = broad-exception-caught
        # An exception occurred during decoding, so it's not base64 encoded
        return False


def ingest_docs(data_dir: str, filename: str) -> None:
    """Ingest documents to the VectorDB."""
    unstruct_reader = download_loader("UnstructuredReader")
    loader = unstruct_reader()
    documents = loader.load_data(file=Path(data_dir), split_documents=False)

    encoded_filename = filename[:-4]
    if not is_base64_encoded(encoded_filename):
        encoded_filename = base64.b64encode(encoded_filename.encode("utf-8")).decode(
            "utf-8"
        )

    for document in documents:
        document.metadata = {"filename": encoded_filename}

    index = get_vector_index()

    node_parser = SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)
    index.insert_nodes(nodes)
