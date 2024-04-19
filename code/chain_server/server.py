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

"""The definition of the Llama Index chain server."""
import base64
import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from chain_server import chains

# create the FastAPI server
app = FastAPI()
# prestage the embedding model
_ = chains.get_embedding_model()
# set the global service context for Llama Index
chains.set_service_context("local", "playground_mistral_7b", "10.123.45.678", 256, 0.7)


class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    question: str
    context: str
    use_knowledge_base: bool = True
    num_tokens: int = 50
    inference_mode: str
    local_model_id: str
    nvcf_model_id: str
    nim_model_ip: str
    nim_model_id: str
    temp: float


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    content: str
    num_docs: int = 4


@app.get("/health")
async def health() -> JSONResponse:
      return JSONResponse(
        content={"status": "OK"}, status_code=200
    )


@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document to the vector store."""
    if not file.filename:
        return JSONResponse(content={"message": "No files provided"}, status_code=200)

    upload_folder = tempfile.mkdtemp()
    upload_file = os.path.basename(file.filename)
    if not upload_file:
        raise RuntimeError("Error parsing uploaded filename.")
    file_path = os.path.join(upload_folder, upload_file)
    uploads_dir = Path(upload_folder)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chains.ingest_docs(file_path, upload_file)

    return JSONResponse(
        content={"message": "File uploaded successfully"}, status_code=200
    )

@app.post("/generate")
async def generate_answer(prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""
    
    if prompt.use_knowledge_base:
        generator = chains.rag_chain_streaming(prompt.question, 
                                               prompt.num_tokens, 
                                               prompt.inference_mode, 
                                               prompt.local_model_id,
                                               prompt.nvcf_model_id,
                                               prompt.nim_model_ip,
                                               prompt.nim_model_id,
                                               prompt.temp)
        return StreamingResponse(generator, media_type="text/event-stream")  

    generator = chains.llm_chain_streaming(prompt.context, 
                                           prompt.question, 
                                           prompt.num_tokens, 
                                           prompt.inference_mode, 
                                           prompt.local_model_id,
                                           prompt.nvcf_model_id,
                                           prompt.nim_model_ip,
                                           prompt.nim_model_id,
                                           prompt.temp)
    return StreamingResponse(generator, media_type="text/event-stream")    


@app.post("/documentSearch")
def document_search(data: DocumentSearch) -> List[Dict[str, Any]]:
    """Search for the most relevant documents for the given search parameters."""
    retriever = chains.get_doc_retriever(num_nodes=data.num_docs)
    nodes = retriever.retrieve(data.content)
    output = []
    for node in nodes:
        file_name = nodes[0].metadata["filename"]
        decoded_filename = base64.b64decode(file_name.encode("utf-8")).decode("utf-8")
        entry = {"score": node.score, "source": decoded_filename, "content": node.text}
        output.append(entry)

    return output
