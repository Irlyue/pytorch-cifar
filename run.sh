#!/usr/bin/env bash

set -e

export n_classes=10
export batch_size=128
export model_dir=~/tmp/cifar10


if [[ "$1" == "train" ]]; then
    export n_epochs=20
    export CUDA_VISIBLE_DEVICES='6'

    export lr=1e-1
    python train.py
    export lr=1e-2
    python train.py
    export lr=1e-3
    python train.py
elif [[ "$1" == "delete" ]]; then
    echo "Removing ${model_dir}"
    rm -rI ${model_dir}
elif [[ "$1" == "eval" ]]; then
    python evaluate.py
fi
