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
import requests
import mimetypes

import hashlib
import json
import os

# Function to check if running in a Jupyter notebook
def in_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        return False

class DocProcessor:
    def __init__(self, docs_dir, mount_docs_dir, upload_url, log=False):
        self.docs_dir = docs_dir
        self.mount_docs_dir = mount_docs_dir
        self.upload_url = upload_url
        self.record_file = os.path.join(docs_dir, ".file_cache.json")
        self.record_lock_file = os.path.join(docs_dir, ".file_cache.lock")
        self.record = dict()
        self.log = log

    def _calculate_hash(self, filepath):
        """Calculate the SHA256 hash of the file content."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _save(self):
        """Save the processed file's hash to the disk and release the lock."""
        with open(self.record_file, "w") as file:
            json.dump(self.record, file)
        
        os.remove(self.record_lock_file)
    
    def _load(self):
        """Load the processed file's hash from the if not locked disk."""
        if os.path.exists(self.record_lock_file):
            raise FileExistsError(f"Cache file is locked. Another process is likely indexing docs. If not, delete {self.record_lock_file}")
        
        with open(self.record_lock_file, "w") as file:
            file.write("LOCKED")

        if os.path.exists(self.record_file):
            with open(self.record_file, "r") as file:
                self.record = json.load(file)

    def _has_been_processed(self, filepath):
        """Check if the file has been processed."""
        filehash = self._calculate_hash(filepath)
        return self.record.get(filepath) == filehash, filehash

    def _process_doc(self, filepath):
        """Process the file if it hasn't been processed or if the content has changed."""
        has_been_processed, filehash = self._has_been_processed(filepath)
        if not has_been_processed:
            try:
                self._upload_document(filepath)
                self.record[filepath] = filehash

                if self.log:
                    print(f"{filepath} - processed.")
            except Exception as err:
                    print(f"{filepath} - failed: {err}")
        else:
            if self.log:
                print(f"Skipping {filepath}, already processed.")

    def _upload_document(self, file_path):
        headers = {
            'accept': 'application/json'
        }
        mime_type, _ = mimetypes.guess_type(file_path)
        files = {
            'file': (file_path, open(file_path, 'rb'), mime_type)
        }
        response = requests.post(self.upload_url, headers=headers, files=files)

        if response.status_code != 200:
            raise Exception(f"Document upload failed with status code {response.status_code}: {response.text}")
        
        if "File uploaded successfully" not in response.text:
            raise Exception(f"Document upload failed: {response.text}")
    
    def _count_files(self, dir_list):
        file_count = 0
        for directory in dir_list:
            for _, _, files in os.walk(directory):
                file_count += len(files)
        return file_count
    
    def process(self):
        if in_jupyter():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # load cache data and lock
        self._load()

        dir_list = [self.docs_dir, self.mount_docs_dir]
        total_files = self._count_files(dir_list)

        try:
            with tqdm(total=total_files) as pbar:
                for directory in dir_list:
                    for root, _, files in os.walk(directory):
                        for file in files:
                            if file in [".gitkeep", ".file_cache.json", ".file_cache.lock"]:
                                pbar.update(1)
                                continue
                                
                            file_path = os.path.join(root, file)
                            self._process_doc(file_path)
                            pbar.update(1)
        finally:
            # save cache data and unlock
            self._save()


if __name__ == "__main__":
    p = DocProcessor("../data/documents", "/mnt/docs", "http://localhost:8000/uploadDocument", True)
    p.process()