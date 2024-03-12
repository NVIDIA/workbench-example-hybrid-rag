# This script stops the local TGI Inference Server. 

pkill -SIGINT -f "^text-generation-server download-weights" & 
pkill -SIGINT -f '^text-generation-launcher' & 
pkill -SIGINT -f 'text-generation' & 

sleep 1
exit 0