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

TEXT_SPLITTER_MODEL = "intfloat/e5-large-v2"
TEXT_SPLITTER_CHUNCK_SIZE = 510
TEXT_SPLITTER_CHUNCK_OVERLAP = 200
EMBEDDING_MODEL = "intfloat/e5-large-v2"
DEFAULT_NUM_TOKENS = 50
DEFAULT_MAX_CONTEXT = 800

LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest assistant."
    "Always answer as helpfully as possible, while being safe."
    "Please ensure that your responses are positive in nature."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)

LLAMA_RAG_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "Use the following context to answer the user's question. If you don't know the answer,"
    "just say that you don't know, don't try to make up an answer."
    "<</SYS>>"
    "<s>[INST] Context: {context_str} Question: {query_str} Only return the helpful"
    " answer below and nothing else. Helpful answer:[/INST]"
)


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
def get_llm(inference_mode: str, nvcf_model_id: str, nim_model_ip: str, num_tokens: int, temp: float) -> LangChainLLM:
    """Create the LLM connection."""
    
    if inference_mode == "local":
        inference_server_url_local = "http://127.0.0.1:9090/"

        llm_local = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            max_new_tokens=num_tokens,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=temp,
            repetition_penalty=1.03,
            streaming=True
        )
    else: 
        inference_server_url_local = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else "http://" + nim_model_ip + ":9999/v1/"
        
        llm_local = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_local,
            max_new_tokens=num_tokens,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=temp,
            repetition_penalty=1.03,
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
def set_service_context(inference_mode: str, nvcf_model_id: str, nim_model_ip: str, num_tokens: int, temp: float) -> None:
    """Set the global service context."""
    service_context = ServiceContext.from_defaults(
        llm=get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp), embed_model=get_embedding_model()
    )
    set_global_service_context(service_context)

def llm_chain_streaming(
    context: str, 
    question: str, 
    num_tokens: int, 
    inference_mode: str, 
    nvcf_model_id: str, 
    nim_model_ip: str,
    nim_model_id: str,
    temp: float,
) -> Generator[str, None, None]:
    """Execute a simple LLM chain using the components defined above."""
    set_service_context(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp)

    if inference_mode == "local":
        prompt = LLAMA_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        response = get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp).stream_complete(prompt, max_new_tokens=num_tokens)
        gen_response = (resp.delta for resp in response)
        for chunk in gen_response:
            yield chunk
    else:
        openai.api_key = os.environ.get('NVCF_RUN_KEY') if inference_mode == "cloud" else "xyz"
        openai.base_url = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else "http://" + nim_model_ip + ":9999/v1/"
        prompt = LLAMA_CHAT_TEMPLATE.format(context_str=context, query_str=question)
        
        completion = openai.chat.completions.create(
          model= nvcf_model_id if inference_mode == "cloud" else nim_model_id,
          temperature=temp,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens, 
          stream=True
        )
        
        for chunk in completion:
            yield chunk.choices[0].delta.content

def rag_chain_streaming(prompt: str, 
                        num_tokens: int, 
                        inference_mode: str, 
                        nvcf_model_id: str, 
                        nim_model_ip: str,
                        nim_model_id: str,
                        temp: float) -> "TokenGen":
    """Execute a Retrieval Augmented Generation chain using the components defined above."""
    set_service_context(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp)

    if inference_mode == "local":
        get_llm(inference_mode, nvcf_model_id, nim_model_ip, num_tokens, temp).llm.max_new_tokens = num_tokens  # type: ignore
        nodes = get_doc_retriever(num_nodes=2).retrieve(prompt)
        docs = []
        for node in nodes: 
            docs.append(node.get_text())
        prompt = LLAMA_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        response = get_llm(inference_mode, 
                           nvcf_model_id, 
                           nim_model_ip, 
                           num_tokens, 
                           temp).stream_complete(prompt, max_new_tokens=num_tokens)
        gen_response = (resp.delta for resp in response)
        for chunk in gen_response:
            yield chunk
    else: 
        openai.api_key = os.environ.get('NVCF_RUN_KEY') if inference_mode == "cloud" else "xyz"
        openai.base_url = "https://integrate.api.nvidia.com/v1/" if inference_mode == "cloud" else "http://" + nim_model_ip + ":9999/v1/"
        num_nodes = 1 if ((inference_mode == "cloud" and nvcf_model_id == "playground_llama2_13b") or (inference_mode == "cloud" and nvcf_model_id == "playground_llama2_70b")) else 2
        nodes = get_doc_retriever(num_nodes=num_nodes).retrieve(prompt)
        docs = []
        for node in nodes: 
            docs.append(node.get_text())
        prompt = LLAMA_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
        
        completion = openai.chat.completions.create(
          model=nvcf_model_id if inference_mode == "cloud" else nim_model_id,
          temperature=temp,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=num_tokens,
          stream=True
        )
        for chunk in completion:
            yield chunk.choices[0].delta.content

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
