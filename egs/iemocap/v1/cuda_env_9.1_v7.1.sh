#!/bin/bash
# originally from nanxin but the CPATH was changed (order of $CPATH and the paths specified here)
#export LD_LIBRARY_PATH=/export/b14/nchen/nccl_2.1.4-1+cuda8.0_x86_64/lib:/export/b18/nchen/libgpuarray/lib:/usr/local/cuda/lib64:/export/b18/nchen/cuda/lib64:/export/b18/nchen/cuda/include:/export/b18/nchen/mpi/lib:~/.local/lib:$LD_LIBRARY_PATH
#export CPATH=/export/b14/nchen/nccl_2.1.4-1+cuda8.0_x86_64/include:/export/b18/nchen/cuda/lib64:/export/b18/nchen/cuda/include:~/.local/include:/export/b18/nchen/libgpuarray/src:$CPATH
#export LIBRARY_PATH=/export/b14/nchen/nccl_2.1.4-1+cuda8.0_x86_64/lib:/export/b18/nchen/cuda/lib64:/export/b18/nchen/cuda/include:~/.local/lib:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda/bin/:/export/b18/nchen/mpi/bin:/export/b18/nchen/cntk/bin:$PATH
#export C_INCLUDE_PATH=/export/b18/nchen/libgpuarray/src:$C_INCLUDE_PATH

# cuda
CUDAROOT=/usr/local/cuda
export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$CUDAROOT/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDAROOT/lib64:$LIBRARY_PATH
#export CPATH=$CUDAROOT/include:$CPATH # only jesus included this but there is no file actually
#temporarily for warp-ctc installation with gpu support
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

# cudnn
#export LD_LIBRARY_PATH=/home/jcho/cudnn/cudnn-9.1-v7.1/lib64:$LD_LIBRARY_PATH
#export CPATH=/home/jcho/cudnn/cudnn-9.1-v7.1/include:$CPATH
#export LIBRARY_PATH=/home/jcho/cudnn/cudnn-9.1-v7.1/lib64:$LIBRARY_PATH

#export LD_LIBRARY_PATH=/home/jcho/cudnn/cudnn-8.0-v6.0/lib64:$LD_LIBRARY_PATH
#export CPATH=/home/jcho/cudnn/cudnn-8.0-v6.0/include:$CPATH
#export LIBRARY_PATH=/home/jcho/cudnn/cudnn-8.0-v6.0/lib64:$LIBRARY_PATH
