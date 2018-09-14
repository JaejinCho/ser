# Throw gpu-using jobs with this script ex) qsub -e LOG -o LOG -cwd -l
    # mem_free=2G,ram_free=2G,gpu=1(,hostname=b17) ./sub.sh pythonscript.py
# $@ is all of the parameters passed to the script. For instance, if you call
# ./someScript.sh foo bar then $@ will be equal to foo bar
unset PYTHONPATH
source activate py3_conda9.1
#source /export/b17/jcho/emotion_recognition/IEMOCAP/v2_20180307/script/cuda_env_9.1_v7.1.sh
source cuda_env_9.1_v7.1.sh
# JJ: for h5py (hdf5 realted thing)
export CPATH=$HOME/myapps/include:$CPATH
export LD_LIBRARY_PATH=$HOME/myapps/lib:$LD_LIBRARY_PATH
# gpu running config
#THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python "$@"
#THEANO_FLAGS='floatX=float32,device=cuda,gpuarray.preallocate=1' python "$@"
CUDA_VISIBLE_DEVICES=`free-gpu` python "$@"
