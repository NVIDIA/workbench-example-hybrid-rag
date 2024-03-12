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
import shutil
from milvus import default_server
from pymilvus import connections, utility, Collection

# Get connection to DB
connections.connect(host='localhost', port=19530)

# Get collection
collection = Collection(utility.list_collections()[0])

# Delete entities
expr = "id!=''"
collection.delete(expr)

# Query all entities
result = collection.query(expr="id!=''", output_fields=["id"])

# Extract all IDs
id_array = [entity["id"] for entity in result]

# Ensure database is empty
print(id_array)

if os.path.exists("/project/data/documents/.file_cache.json"):
    os.remove("/project/data/documents/.file_cache.json")
    
if os.path.exists("/project/data/documents/.file_cache.lock"):
    os.remove("/project/data/documents/.file_cache.lock")

if os.path.exists("/project/data/documents/.ipynb_checkpoints"):
    shutil.rmtree("/project/data/documents/.ipynb_checkpoints")