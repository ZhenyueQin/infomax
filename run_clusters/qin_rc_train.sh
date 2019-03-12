#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=15GB

module load cudnn/v6
module load tensorflow/1.4.0-py36-gpu
#module load keras/2.2.4-py36
#module load torchvision/0.2.1-py36

#module load opencv/2.4.13.2
python infomax_tf_keras_launcher.py --mission train_a_model
