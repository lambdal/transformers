#!/bin/bash

GPU_NAME=${1:-"hyperplanebasic"}
PATH_PTXAS=${2:-"/usr/bin/ptxas"}
PATH_LIBDEVICE=${3:-"/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc"}

mkdir -p $GPU_NAME

echo $PATH_PTXAS
echo $PATH_LIBDEVICE

python examples/benchmarks.py --torch --torch_cuda --save_to_csv --csv_filename="$/logs/{GPU_NAME}/inference_pytorch_cu_fp32.txt" 


python examples/benchmarks.py --torch --torch_cuda --fp16 --save_to_csv --csv_filename="$/logs/{GPU_NAME}/inference_pytorch_cu_fp16.txt"


python examples/benchmarks.py --tensorflow --save_to_csv --csv_filename="/logs/${GPU_NAME}/inference_tensorflow_fp32.txt"


mkdir -p bin
cp $PATH_PTXAS ./bin 

mkdir -p ~/xla/nvvm/libdevice
cp $PATH_LIBDEVICE ~/xla/nvvm/libdevice

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/${USER}/xla"

python examples/benchmarks.py --tensorflow --xla --save_to_csv --csv_filename="$/logs/{GPU_NAME}/inference_tensorflow_xla_fp32.txt"


python examples/benchmarks.py --tensorflow --amp --save_to_csv --csv_filename="$/logs/{GPU_NAME}/inference_tensorflow_amp.txt"