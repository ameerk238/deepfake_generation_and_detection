#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  apt-get update
  python3 -m pip install --upgrade pip
  apt-get update && apt-get install -y python3-opencv
  python3 -m pip install tensorboard
  python3 -m pip install moviepy
  python3 -m pip install av
  python3 -m pip install tqdm
  python3 -m pip install lpips


     
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi