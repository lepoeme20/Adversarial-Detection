#!/bin/bash
# === Set parameters ===
## TODO: Low CW (lr=0.015, k=1), Low PGD-1 (1/255), Low PGD-2 (2/255)
detectors=('lid')
attacks=('PGD') # 'FGSM' 'PGD' 'BIM' 'CW' 'DF' 'FAB' 'PGD2' 'AUTO' 'SQUARE'
datasets=('svhn' 'cifar100' 'cifar10') # 'svhn' 'cifar100' 'cifar10'
networks=('resnet' 'vgg') # 'resnet' 'vgg' 'custom'
gpu_id=(0)

for detector in ${detectors[@]}
do
    for network in ${networks[@]}
    do
        for dataset in ${datasets[@]}
        do
            for attack in ${attacks[@]}
            do
            start_time=`date +%s`
            if [ $detector = 'lid' ] ; then
                echo "-------------------------------------------------------------------------"
                echo "Detector: LID"
                echo "Baseline: $network"
                echo "Dataset: $dataset"
                echo "Attack: $attack"
                echo "-------------------------------------------------------------------------"
                echo ""
                python detect_lid.py --dataset $dataset --device-ids ${gpu_id[@]} \
                --attack-name $attack --model $network --batch-size 256 --eps 8/255
                echo "-------------------------------------------------------------------------"
                end_time=`date +%s`
                echo execution time was `expr $end_time - $start_time` s.
                echo ""
            fi
            done
        done
    done
done
