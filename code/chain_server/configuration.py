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

"""The definition of the application configuration."""
from chain_server.configuration_wizard import ConfigWizard, configclass, configfield


@configclass
class MilvusConfig(ConfigWizard):
    """Configuration class for the Weaviate connection.

    :cvar url: URL of Milvus DB
    """

    url: str = configfield(
        "url",
        default="http://localhost:9091",
        help_txt="The host of the machine running Milvus DB",
    )


# @configclass
# class TritonConfig(ConfigWizard):
#     """Configuration class for the Triton connection.

#     :cvar server_url: The location of the Triton server hosting the llm model.
#     :cvar model_name: The name of the hosted model.
#     """

#     server_url: str = configfield(
#         "server_url",
#         default="localhost:8001",
#         help_txt="The location of the Triton server hosting the llm model.",
#     )
#     model_name: str = configfield(
#         "model_name",
#         default="ensemble",
#         help_txt="The name of the hosted model.",
#     )


@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar milvus: The configuration of the Milvus vector db connection.
    :type milvus: MilvusConfig
    :cvar triton: The configuration of the backend Triton server.
    :type triton: TritonConfig
    """

    milvus: MilvusConfig = configfield(
        "milvus", env=False, default="http://127.0.0.1:19530", help_txt="The configuration of the Milvus connection."
    )
    # triton: TritonConfig = configfield(
    #     "triton",
    #     env=False,
    #     help_txt="The configuration for the Triton server hosting the embedding models.",
    # )
