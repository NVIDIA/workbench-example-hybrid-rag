# A Hybrid RAG Project on AI Workbench
This is an [NVIDIA AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) project for developing a [Retrieval Augmented Generation](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) application with a customizable Gradio Chat app. It lets you:
* Embed your documents into a locally running vector database.
* Run inference in the **cloud** using NVIDIA inference endpoints for the following supported models:
    * Mistral 7B
    * Mixtral 8x7B
    * Llama 2 13B
    * Llama 2 70B
* Run inference **locally** utilizing a local Hugging Face TGI server for the following supported models:
    * Mistral 7B (4-bit, 8-bit, and no quantization)
    * Llama 2 7B (4-bit, 8-bit, and no quantization)
    * Other models may be specified using their Hugging Face tag
* Run inference with a self-hosted **microservice** using [NVIDIA Inference Microservices (NIMs)](https://www.nvidia.com/en-us/ai/) on any NIM-supported models.
    * Remote microservice
    * Local microservice (Docker users only)

# Quickstart
This section demonstrates how to use this project to run RAG using inference via NVIDIA cloud endpoints. 

### Prerequisites
- An [NGC account](https://ngc.nvidia.com/signin) is required to generate an NVCF run key. 
- A valid NVCF key is required to access NVIDIA API endpoints. Generate a key [here](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2) by clicking "Get API Key". 
- A Hugging Face API token is recommended for access locally-running Llama 2 models. [See how to create one here](https://huggingface.co/docs/hub/en/security-tokens).

### Tutorial: Using a Cloud Endpoint

<img src="./code/chatui/static/cloud.gif" width="66%" height="auto">

1. [Install and configure](#nvidia-ai-workbench) AI Workbench locally and open up AI Workbench. Select a location of your choice. 
2. Fork this repo into *your own* GitHub account.
3. **Inside AI Workbench**:
    - Click **Clone Project** and enter the repo URL of your newly-forked repo.
    - AI Workbench will automatically clone the repo and build out the project environment, which can take several minutes to complete. 
    - Upon `Build Complete`, select **Open Chat** by clicking the green button at the top right. 
    - When prompted, enter your **Hugging Face token** and **NVIDIA NVCF run key** as project secrets.
        - There is a known issue for build 0.44.8 where the secret(s) may truncate when pasted. Alternatively, configure the secret by ``Environment`` > ``Secrets`` > ``<secret_name>`` > ``Configure``. 
    - Select **Open Chat**, and the Gradio app will open in a browser. This takes around 30 seconds.
4. **In the Gradio Chat app**:
    - Click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
    - Select the **Cloud** option and submit a query. 
    - To perform RAG, select **Upload Documents Here** from the right hand panel of the chat UI.
         - You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
    - When the database starts, select **Click to Upload** and choose the text files to upload.
    - Once the files upload, the **Toggle to Use Vector Database** next to the text input box will turn on.
    - Now query your documents! What are they telling you?
    - To change the endpoint, select a different model from the right-hand dropdown and continue querying.

---
**Next Steps:**
* If you get stuck, check out the ["Troubleshooting"](#troubleshooting) section.
* For tutorials on other supported inference modes, check out the ["Advanced Tutorials"](#advanced-tutorials) section below. 

---

### NVIDIA AI Workbench
**Note:** [NVIDIA AI Workbench](https://www.youtube.com/watch?v=ntMRzPzSvM4) is the easiest way to get this RAG app running.
- NVIDIA AI Workbench is a <ins>free client application</ins> that you can install on your own machines.
- It provides portable and reproducible dev environments by handling Git repos and containers for you.
- Installing on a local system? Check out our guides here for [Windows](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/windows.html), [Local Ubuntu 22.04](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-local.html) and for [macOS 12 or higher](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/macos.html)
- Installing on a remote system? Check out our guide for [Remote Ubuntu 22.04](https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-remote.html)

## Troubleshooting

### How do I open AI Workbench?
- Make sure you [installed](#nvidia-ai-workbench) AI Workbench. There should be a desktop icon on your system. Double click it to start AI Workbench.

    <img src="./code/chatui/static/desktop-app.png" width="10%" height="auto">

### How do I clone this repo with AI Workbench?
- Make sure you have opened AI Workbench.
- Click on the **Local** location (or whatever location you want to clone into).
- If this is your first project, click the green **Clone Existing Project** button.
    - Otherwise, click **Clone Project** in the top right
- Drop in the repo URL, leave the default path, and click **Clone**. 

    <img src="./code/chatui/static/clone.png" width="66%" height="auto">

### I've cloned the project, but now nothing seems to be happening?
- The container is building and can take several minutes.
- Look at the very <ins>bottom</ins> of the Workbench window, you will see a **Build Status** widget.
- Click it to expand the build output. 
- When the container is built, the widget will say `Build Ready`.
- Now you can begin. 

    <img src="./code/chatui/static/built.png" width="66%" height="auto">

### How do I start the Chat application?
- Check that the container finished building.
- When it finishes, click the green **Open Chat** button at the top right.

    <img src="./code/chatui/static/chat.png" width="66%" height="auto">

### How can I customize this project with AI Workbench?
- Check that the container is built.
- Then click the green **dropdown** next to the `Open Chat` button at the top right.
- Select **JupyterLab** to start editing the code. Alternatively, you may configure VSCode support [here](https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications/built-in/vs-code.html).

    <img src="./code/chatui/static/jupyter.png" width="66%" height="auto">

# Advanced Tutorials
This section shows you how to use difference inference modes with this RAG project. For these tutorials, a GPU of at least 12 GB of vRAM is recommended. If you don't have one, go back to the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint) that shows how to use **Cloud Endpoints**. 

## Tutorial 1: Using a local GPU
This tutorial assumes you already cloned this Hybrid RAG project to your AI Workbench. If not, please follow the beginning of the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint). 

<img src="./code/chatui/static/local.gif" width="66%" height="auto">

**Inference**

1. Select the green **Open Chat** button on the top right the AI Workbench project window. 
    * You may be prompted to enter your NVCF and Hugging Face keys as project secrets. If so, do it and then select **Open Chat** again. If not, you have entered them previously. 
    * There is a known issue for build 0.44.8 where the secret(s) may truncate when pasted. Alternatively, you can configure the secret by ``Environment`` > ``Secrets`` > ``<secret_name>`` > ``Configure``. 
2. Once the UI opens, click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
3. Select the **Local System** inference mode under ``Inference Settings`` > ``Inference Mode``. 
4. Select a model from the dropdown on the right hand settings panel. Mistral 7B and Llama 2 are currently supported as default.
    * ``Mistral 7B``: This model is ungated and is easiest to use.
    * ``Llama 2 7B``: This model is gated. Ensure the Hugging Face API Token is configured properly. You can configure the secret by ``Environment`` > ``Secrets`` > ``<secret_name>`` > ``Configure`` and restart the environment if needed. 
    * You can also input a custom model from Hugging Face, following the same format. Careful--not all models and quantization levels are supported in this RAG!
5. Select a quantization level. Full, 8-bit, and 4-bit bitsandbytes precision levels are currently supported. 

##### Table 1 System Resources vs Model Size and Quantization

    | vRAM    | System RAM | Disk Storage | Model Size & Quantization |
    |---------|------------|--------------|---------------------------|
    | >=12 GB | 32 GB      | 40 GB        | 7B & int4                 |
    | >=24 GB | 64 GB      | 40 GB        | 7B & int8                 |
    | >=40 GB | 64 GB      | 40 GB        | 7B & none                 |

6. Select **Load Model** to pre-fetch the model. This will take up to several minutes to perform an initial download of the model to the project cache. Subsequent loads will detect this cached model. 
7. Select **Start Server** to start the inference server with your current local GPU. This may take a moment to warm up.
8. Now, start chatting! Queries will be made to the model running on your local system whenever this inference mode is selected.

**Using RAG**

9. In the right hand panel of the Chat UI select **Upload Documents Here**. Click to upload or drag and drop the desired text files to upload.
   * You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
10. Once the files upload, the **Toggle to Use Vector Database** next to the text input box will turn on by default.
11. Now query your documents! To use a different model, stop the server, make your selections, and restart the inference server. 

## Tutorial 2: Using a Remote Microservice
This tutorial assumes you already cloned this Hybrid RAG project to your AI Workbench. If not, please follow the beginning of the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint). 

<img src="./code/chatui/static/microservice.gif" width="75%" height="auto">

**Prerequisites**

* Set up your NVIDIA NeMo Inference Microservice to run self-hosted on another system of your choice. After joining the [EA Program](https://developer.nvidia.com/nemo-microservices-early-access), the playbook to get started is located [here](https://developer.nvidia.com/docs/nemo-microservices/inference/nmi_playbook.html). Remember the _model name_ and the _ip address_ of this running microservice. 

**Inference**

1. Select the green **Open Chat** button on the top right the AI Workbench project window. 
    * You may be prompted to enter your NVCF and Hugging Face keys as project secrets. If so, do it and then select **Open Chat** again. If not, you have entered them previously. 
    * There is a known issue for build 0.44.8 where the secret(s) may truncate when pasted. Alternatively, you can configure the secret by ``Environment`` > ``Secrets`` > ``<secret_name>`` > ``Configure``. 
2. Once the UI opens, click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
3. Select the **Self-hosted Microservice** inference mode under ``Inference Settings`` > ``Inference Mode``. 
4. Select the **Remote** tab in the right hand settings panel. Input the **IP address** of the system running the microservice, as well as the **model name** selected to run with that microservice. 
5. Now start chatting! Queries will be made to the microservice running on a remote system whenever this inference mode is selected.

**Using RAG**

6. In the right hand panel of the Chat UI select **Upload Documents Here**. Click to upload or drag and drop the desired text files to upload. 
   * You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
7. Once uploaded successfully, the **Toggle to Use Vector Database** should turn on by default next to your text input box.
8. Now you may query your documents!

## Tutorial 3: Using a Local Microservice

#### If you don't have Docker experience, don't try this section. If you do have some Docker experience, it should be fairly straight forward.

Spinning up a Microservice locally from inside the AI Workbench Hybrid RAG project is an area of active development. This tutorial has been tested on 1x RTX 4090 and is currently being improved. 

Here are some important **PREREQUISITES**:
* This tutorial assumes you already have this Hybrid RAG project cloned to your AI Workbench. If not, please first follow steps 1-5 of the project [Quickstart](#quickstart). 
* Your AI Workbench must be running with a **DOCKER** container runtime. Podman is currently unsupported.
* You must already be accepted into the NeMo Inference Microservice [EA Program](https://developer.nvidia.com/nemo-microservices-early-access). 
* You must have generated your own TRT-LLM model engine files in some model store directory located on your local system. These are models you would like to serve for inference.
* Shut down any locally-running inference servers (eg. from Tutorial 1), as these may result in memory issues when running the microservice locally. 

**Inference**

1. In the AI Workbench project window, navigate to <inx>Environment</ins> &#8594; <ins>Mounts</inx> &#8594; <ins>Add</ins>. Add the following host mount:
    * _Type_: Host Mount
    * _Target_: ``/opt/host-run``
    * _Source_: ``/var/run``
    * _Description_: Mount for Docker socket (NIM on Local RTX)
2. Navigate to <ins>Environment</ins>&#8594;<ins>Secrets</ins>. Configure the existing secrets and create a new secret with the following details.
    * _Name_: NGC_CLI_API_KEY
    * _Value_: (Your NGC API Key)
    * _Description_: NGC API Key for NIM access
3. Navigate to <ins>Environment</ins>&#8594;<ins>Variables</ins>. Ensure the following are configured. Restart your environment if needed. 
    * DOCKER_HOST: location of your docker socket, eg. ``unix:///opt/host-run/docker.sock``
    * LOCAL_NIM_MODEL_STORE: location of your ``model-store`` directory, eg. ``/mnt/c/Users/NVIDIA/model-store``
4. Open the Chat application from the AI Workbench project window. 
    * You may be prompted to enter your NVCF and Hugging Face keys as project secrets. You may do so, and then select **Open Chat** again.
    * If you are given no prompt, you may have already entered the keys before. You may find them in AI Workbench under Environment&#8594;Secrets.
5. Once the UI opens, select the **Self-hosted Microservice** inference mode under Inference Settings &#8594; Inference Mode. Wait for the RAG backend to start up, which may take a few moments.
6. Select the **Local (RTX)** tab in the right hand settings panel. Input the model name of your TRT-LLM engine file. Select **Start Microservice Locally**. This may take a few moments to complete. 
7. Now, you can start chatting! Queries will be made to your microservice running on the local system whenever this inference mode is selected.

**Using RAG**

8. To perform RAG, select **Upload Documents Here** from the right hand panel of the chat UI. Select **Update Database** and choose the text files to upload. 
9. Once uploaded successfully, the **Toggle to Use Vector Database** should turn on by default next to your text input box.
10. Now you may query your documents!

## Tutorial 4: Customizing the Gradio App
By default, you may customize Gradio app using the jupyterlab container application. Alternatively, you may configure VSCode support [here](https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications/built-in/vs-code.html).

1. In AI Workbench, select the green **dropdown** from the top right and select **Open JupyterLab**.
2. Go into the `code/chatui/` folder and start editing the files.
3. Save the files.
4. To see your changes, stop the Chat UI and restart it.
5. To version your changes, commit them in the Workbench project window and push to your GitHub repo.

In addition to modifying the Gradio frontend, you can also use the Jupyterlab or another IDE to customize other aspects of the project, eg. custom chains, backend server, scripts, etc.

## License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/nv-edwli/workbench-example-hybrid-rag/blob/main/LICENSE.txt)

This project may download and install additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 
