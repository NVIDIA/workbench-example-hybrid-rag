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

import os
import torch

def check_setup():
    inference_mode = os.environ.get("INFERENCE_MODE")
    model_id = os.environ.get("MODEL_ID")
    quantize = os.environ.get("QUANTIZE")
    num_shard = os.environ.get("NUM_SHARD")
    embedding_device = os.environ.get("EMBEDDING_DEVICE")

    if inference_mode == "local":
        if os.environ.get("HUGGING_FACE_HUB_TOKEN") is None:
            print("WARNING: You must set your HUGGING_FACE_HUB_TOKEN to download Llama2 when running locally.")
        
        if model_id is None:
            print("WARNING: You must set a model ID to run local inference (e.g. meta-llama/Llama-2-7b-chat-hf)")

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("WARNING: You do not have any GPUs available. Local inference will fail. Double check your host and the 'Hardware' section on the Environment tab of this project in AI Workbench.")

        elif num_gpus == 1:
            if embedding_device != "cpu":
                print("WARNING: You only have 1 GPU available that will be used for inference. You must set the embedding device to 'cpu'.")
                    
            if num_shard is not None and num_shard > 1:
                print("WARNING: You cannot shard with only 1 GPU available.")
        else:
            if num_shard is not None:
                if num_shard >= num_gpus:
                    print("WARNING: You should reserve the last GPU for embedding. The value for `NUM_SHARD` should be at least 1 less than the total available GPUs")
            
        if embedding_device != "cpu" and embedding_device != f"cuda:{num_gpus-1}":
            print("WARNING: You should set the embedding device to your last GPU. When the inference server starts it will start with the first GPU. Make sure your sharding and quantization settings work for your hardware configuration.")

        inf_gpu_cnt = num_shard if num_shard is not None else 1
        inf_mem = 0
        for i in range(inf_gpu_cnt):
            inf_mem += torch.cuda.get_device_properties(i).total_memory
        
        print("Inference Mode: Local")
        print(f"Num GPUs: {num_gpus}")
        print(f"Num Inference GPUs: {inf_gpu_cnt}")
        print(f"Total Inference VRAM: {inf_mem/(2**30):.2f} GB")
        print(f"Quantize Mode: {quantize}")
        print(f"Embedding Device: {embedding_device}")
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            if embedding_device != "cpu":
                print("WARNING: You do not have any GPUs locally. You must set the embedding device to 'cpu'.")
        
        print("Inference Mode: Cloud")
        print(f"Embedding Device: {embedding_device}")
    
    return inference_mode
        