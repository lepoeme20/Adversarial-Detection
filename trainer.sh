#!/bin/bash
# === Set parameters ===
datasets=('cifar10' 'cifar100' 'svhn') # 'cifar10' 'cifar100' 'svhn'
networks=('resnet' 'vgg') # 'resnet' 'vgg'
gpu_id=(0)

for network in ${networks[@]}
do
    for dataset in ${datasets[@]}
    do
        echo "**Save attacked images**"
        echo "-------------------------------------------------------------------------"
        echo "Model: $network"
        echo "Data: $dataset"
        echo "-------------------------------------------------------------------------"
        echo ""
        python trainer.py --dataset $dataset --device-ids ${gpu_id[@]} --model $network
        echo "-------------------------------------------------------------------------"
        echo ""
    done
done
