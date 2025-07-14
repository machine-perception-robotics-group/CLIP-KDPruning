#!/bin/bash
#SBATCH --job-name=Ours0709
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --output=output_dir/x75_Ours0709/Ours0709_result.txt
#SBATCH --error=output_dir/x75_Ours0709/Ours0709_error.txt

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_PORT=29507
export CUDA_VISIBLE_DEVICES=0,1,2,3

singularity exec --nv \
  --env LD_PRELOAD="" \
  --bind /home/miki/UPop/data:/mnt/data \
  kdprune.sif bash -c '

  export CUDA_VISIBLE_DEVICES=0,1,2,3
  
  eval "$(/opt/miniconda/bin/conda shell.bash hook)"
  conda activate upop

  echo "Replacing opencv (conda) with opencv-python-headless (pip)..."
  conda remove -y opencv
  pip install opencv-python-headless

  ### 🚩 追加 : nltkのバージョン固定 + punktだけ入れる
  echo "Installing nltk==3.7 and downloading punkt..."
  pip install "nltk==3.7"
  python3 -m nltk.downloader punkt

  echo "Checking GPU availability..."
  python3 -c "import torch; print(\"CUDA Available:\", torch.cuda.is_available()); print(\"Device:\", torch.cuda.get_device_name(0))"

  echo "Listing data directory:"
  ls /mnt/data

  ### 🚩 分散トレーニングの実行
  python3 -m torch.distributed.run --nproc_per_node=4 --master_port 29507 CLIP_compress.py --p 0.75 --epoch 9 \
  --pretrained pretrained/clip_large_retrieval_coco.pth --config ./configs/retrieval_coco_clip.yaml \
  --output_dir output_dir/x75_Ours0709 --KD --make_sim_matrix > output_dir/x75_Ours0709/Ours0709_log.txt
'
