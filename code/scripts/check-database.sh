# Wait for service to be reachable.
ATTEMPTS=0
MAX_ATTEMPTS=20

while [ $(curl -o /dev/null -s -w "%{http_code}" "http://localhost:19530/v1/vector/collections") -ne 200 ]; 
    do 
      ATTEMPTS=$(($ATTEMPTS+1))
      if [ ${ATTEMPTS} -eq ${MAX_ATTEMPTS} ]
      then
        echo "Max attempts reached: $MAX_ATTEMPTS. Server may have timed out. Stop the container and try again. "
        exit 1
      fi
      
      echo "Polling inference server. Awaiting status 200; trying again in 10s. "
      sleep 10
    done 
