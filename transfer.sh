#!/bin/bash
# === Set parameters ===
# TODO: CIFAR100, SVHN - FAB
detectors=('lid')
sources=('FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE') # 'FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE'
targets=('FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE') # 'FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE'
datasets=('cifar10' 'cifar100' 'svhn') # 'svhn' 'cifar100' 'cifar10'
networks=('resnet' 'vgg') # 'resnet' 'vgg' 'custom'
gpu_id=(0)

for detector in ${detectors[@]}
do
    for network in ${networks[@]}
    do
        for dataset in ${datasets[@]}
        do
            for source in ${sources[@]}
            do
                for target in ${targets[@]}
                do
                if [ $detector = 'lid' ] ; then
                    echo "-------------------------------------------------------------------------"
                    echo "Detector: LID"
                    echo "Baseline: $network"
                    echo "Dataset: $dataset"
                    echo "Source attack: $source"
                    echo "Target attack: $target"
                    echo "-------------------------------------------------------------------------"
                    echo ""
                    python detect_lid_transfer.py --dataset $dataset --device-ids ${gpu_id[@]} \
                    --attack-name $source --model $network --batch-size 256 --target-attack $target \
                    --eps 8/255
                    echo "-------------------------------------------------------------------------"
                    echo ""
                fi
                done
            done
        done
    done
done
