#!/bin/bash 
#SBATCH --array=0
#SBATCH --job-name BP_MC3
#SBATCH --ntasks=1
#SBATCH --partition=RTX3090		## RTX3090, V100-32GB, RTXA6000, A100-PCI, H100 (use double hash as comment) 
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=100G           ## 240G
#SBATCH --time 01-00:00:00
#SBATCH --output console_output/console.%A_%a.out
#SBATCH --error console_output/console.%A_%a.error  ## #SBATCH --time 31-00:00

srun  -K -N1\
	  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.04-py3.sqsh \
	  --container-workdir=`pwd`\
	  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
	  --task-prolog=`pwd`/install.sh \
      --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
	  python3 run_demo.py --model taichi --source_path ./data/taichi/subject1.png --driving_path ./data/taichi/driving1.mp4 ${SLURM_ARRAY_TASK_ID}     
	  
