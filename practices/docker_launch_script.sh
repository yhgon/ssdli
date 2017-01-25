## 50xx for digits port, 100xx for jupyter port for 20xx user id , account for userxx 
PORT=5000
ID_START=2001 ## USER_ID start
ID_OFFSET=$(($(id -u) - $ID_START))
NB_PORT=$((${PORT}*2 + ${ID_OFFSET}+17))  ## offset 17 for server 3, offset 9 for server2 (2001~2008, 2009~2016, 2017~2024)
DIGIT_PORT=$((${PORT} + ${ID_OFFSET}+17))
GPU_ID=$(($ID_OFFSET + $ID_OFFSET))

DL_FRAMEWORK=$1  # caffe, torch, digits, ensorflow 
COMMAND=$2

DATASET=/raid/dataset ## LOCAL raw-data
JOBS=/raid/jobs/$USER ## digits job folder

if [ -z $DL_FRAMEWORK ]
then
    echo 'Please specify which framework you will use (caffe, torch, digits, tensorflow)'
        return
fi

NV_GPU=$ID_OFFSET nvidia-docker run --rm -ti --name=$(id -u) -e NB_UID=$(id -u) -p $NB_PORT:8888 -p $DIGIT_PORT:5000 -v $(pwd):/workspace -v $DATASET:/data -v $JOBS:/jobs dli/$1:17.01 $2

###  NV_GPU for GPU allocate for each 1 gpu
### ti : terminal interactive mode
### --name : use force name instead of random name
### p : port mapping
### -v : mapping storage volume
### /workspace for local_home, digits job folders, dataset folders
