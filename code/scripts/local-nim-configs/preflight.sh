# This script checks for the proper configs for running a NVIDIA Inference Microservice container locally on DOCKER-enabled systems. 

if [ -x "$(command -v docker)" ]; then
    echo "Docker detected. Proceeding with preflight checklist. "
else
    echo "Docker is not detected! Local NIM is currently supported on Docker runtimes only. "
    exit 1
fi

# Preflight checklist
if [[ -z "${NGC_CLI_API_KEY}" ]]; then
  echo "Missing config: the user has not configured their NGC_CLI_API_KEY as a project secret. Can't pull the NIM container!"
  exit 1
elif [[ -z "${DOCKER_HOST}" ]]; then
  echo "Missing config: the user has not configured their DOCKER_HOST as an env variable. See README for details. "
  exit 1
elif [[ -z "${LOCAL_NIM_HOME}" ]]; then
  echo "Missing config: the user has not configured their LOCAL_NIM_HOME as an env variable. See README for details. "
  exit 1
elif [ ! -d "/opt/host-run" ]; then
  echo "Missing config: could not find /opt/host-run. Docker socket mount appears unconfigured. See README for details. "
  exit 1
else
  echo "Configs look good. "
fi

exit 0