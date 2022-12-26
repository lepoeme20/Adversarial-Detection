#!/bin/bash
# === Set parameters ===
# TODO: Low CW (lr=0.015, k=1), Low PGD-1 (1/255), Low PGD-2 (2/255)
attacks=('FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB') # 'FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE'
datasets=('cifar10' 'cifar100' 'svhn') # 'cifar10' 'cifar100' 'svhn'
networks=('resnet' 'vgg') # 'resnet' 'vgg'
gpu_id=(0)

for network in ${networks[@]}
do
    for dataset in ${datasets[@]}
    do
        for attack in ${attacks[@]}
        do
            echo "**Save attacked images**"
            echo "-------------------------------------------------------------------------"
            echo "model: $network"
            echo "Dataset: $dataset"
            echo "Attack: $attack"
            echo "-------------------------------------------------------------------------"
            echo ""
            python get_adv.py --dataset $dataset --device-ids ${gpu_id[@]} \
            --attack-name $attack --model $network --save-adv --eps 8/255
            echo "-------------------------------------------------------------------------"
            echo ""
        done
    done
done
