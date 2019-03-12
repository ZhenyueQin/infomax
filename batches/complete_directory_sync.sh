#!/usr/bin/env bash

#rsync -a --ignore-existing ../. liu162@bracewell.hpc.csiro.au:/home/liu162/Zhenyue/Project-Wukong-VAE-Study/Kristiadi-Generative-Models

rsync -av --update . liu162@bracewell.hpc.csiro.au:/flush2/liu162/Zhenyue-Qin/Project-Hanabi-Information-Bottleneck/Keras-DIM/ --exclude='*.png' --exclude=outs --exclude=generated_outs
