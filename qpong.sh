#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --account=def-josephp
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
QLearningPong.py