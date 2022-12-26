"""Set configuration for the model
"""
import argparse
import multiprocessing
import torch

def str2float(s):
    if '/' in s:
        s1, s2 = s.split('/')
        s = float(s1)/float(s2)
    return float(s)


def parser_setting(parser):
    """Set arguments
    """
    base_args = parser.add_argument_group('base arguments')
    base_args.add_argument(
        '--local_rank', type=int, default=-1, metavar='N', help='Local process rank.'
    )
    base_args.add_argument(
        '--save-path', type=str, default='./baselineCNN/best_model',
        help='save path for best model'
    )
    base_args.add_argument(
        '--workers', type=int, default=multiprocessing.cpu_count()-1, metavar='N',
        help='dataloader threads'
    )
    base_args.add_argument(
        '--padding', type=int, default=4, help='base padding size'
        )
    base_args.add_argument(
        '--img-size', type=int, default=32, help='cropped image size'
        )
    base_args.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'],
        help='Dataset name'
        )
    base_args.add_argument(
        '--model', type=str, default='resnet',
        choices=['resnet', 'vgg', 'custom']
    )
    base_args.add_argument(
        "--data-root-path", type=str, default='/media/data/benchmark', help='data path'
        )
    base_args.add_argument(
        "--n_cpu", type=int, default=multiprocessing.cpu_count(),
        help="number of cpu threads to use during batch generation"
        )
    base_args.add_argument(
        "--device-ids", type=int, nargs='*', default=[0], help="device id"
    )

    trn_args = parser.add_argument_group('training hyper params')
    trn_args.add_argument(
        '--epochs', type=int, default=300, metavar='N',
        help='number of epochs to train (default: auto)'
        )
    trn_args.add_argument(
        '--batch-size', type=int, default=256,
        help='input batch size for training (default: auto)'
        )
    trn_args.add_argument(
        '--test-batch-size', type=int, default=256,
        help='input batch size for testing (default: auto)'
        )
    trn_args.add_argument(
        '--seed', type=int, default=22 #2256, help='Seed for reproductibility'
    )
    trn_args.add_argument(
        '--resume', action='store_true', default=False, help='if resume or not'
    )
    trn_args.add_argument(
        '--resume-model', type=str, default=None, help='resume model path'
    )

    opt_args = parser.add_argument_group('optimizer params')
    opt_args.add_argument(
        '--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: auto)'
        )
    opt_args.add_argument(
        '--b1', type=float, default=0.5, help='momentum (default: 0.9)'
        )
    opt_args.add_argument(
        '--b2', type=float, default=0.99, help='momentum (default: 0.9)'
        )
    oth_args = parser.add_argument_group('others')
    oth_args.add_argument(
        "--sample-interval", type=int, default=1000, help="interval between image samples"
        )
    oth_args.add_argument(
        "--dev-interval", type=int, default=500, help="interval between image samples"
        )

    detector_args = parser.add_argument_group('Detector')
    detector_args.add_argument(
        '--lid-k', default=30, help='the number of k (k-nearnest) in LID'
    )
    attack_args = parser.add_argument_group('Attack')
    # Deeply Supervised Discriminative Learning for Adversarial Defense (baseline)의
    # setting을 최대한 따를 것
    attack_args.add_argument(
        '--save-adv', action='store_true', default=False, help='if save adversarial examples'
    )
    attack_args.add_argument(
        '--attack-name', type=str, default='FGSM',
        choices=['Clean', 'FGSM', 'BIM', 'CW', 'PGD', 'DF', 'FAB', 'SQUARE', 'PGD2', 'AUTO']
    )
    attack_args.add_argument(
        '--target-attack', type=str, default='FGSM',
        choices=['Clean', 'FGSM', 'BIM', 'CW', 'PGD', 'DF', 'FAB', 'SQUARE', 'PGD2', 'AUTO']
    )
    attack_args.add_argument(
        '--eps', type=str2float, default='8/255', help="For bound eta"
    )
    # arguments for PGD
    attack_args.add_argument(
        '--pgd-iters', type=int, default=10, help="# of iteration for PGD attack"
    )
    attack_args.add_argument(
        '--pgd-alpha', type=float, help="Magnitude of perturbation"
    )
    attack_args.add_argument(
        '--pgd-random-start', action='store_true', default=False,
        help="If ture, initialize perturbation using eps"
    )
    # arguments for C&W
    attack_args.add_argument(
        '--cw-c', type=str2float, default=1e-4, help="loss scaler"
    )
    attack_args.add_argument(
        '--cw-kappa', type=float, default=0, help="minimum value on clamping"
    )
    attack_args.add_argument(
        '--cw-iters', type=int, default=100, help="# of iteration for CW grdient descent"
    )
    attack_args.add_argument(
        '--cw-lr', type=float, default=0.1, help="learning rate for CW attack"
    )
    attack_args.add_argument(
        '--cw-targeted', action='store_true', default=False, help="d"
    )
    # arguments for i-FGSM
    attack_args.add_argument(
        '--bim-step', type=int, default=10, help="Iteration for iterative FGSM"
    )
    # argument for DeepFool
    attack_args.add_argument(
        '--df-step', type=int, default=30, help="Iteration for DeepFool"
    )
    attack_args.add_argument(
        '--df-overshoot', type=float, default=0.02, help="parameter for enhancing the noise"
    )
    # argument for FAB
    attack_args.add_argument(
        '--fab-step', type=int, default=30, help="Iteration for DeepFool"
    )
    attack_args.add_argument(
        '--fab-n-restarts', type=int, default=3, help="Iteration for DeepFool"
    )
    return parser

def get_config():
    parser = argparse.ArgumentParser(description="PyTorch Defense by distance-based model")
    default_parser = parser_setting(parser)
    args, _ = default_parser.parse_known_args()
    args.device = torch.device(f'cuda:{args.device_ids[0]}' if torch.cuda.is_available else 'cpu')

    # number of input classes
    # CelebA: Female/Male
    # Cifar100: A hundred classes
    # The rest: Ten classes
    args.num_class = 100 if args.dataset == 'cifar100' else 10
    args.fab_num_class = args.num_class

    # consider low/high confidence in C&W attack
    if args.attack_name.lower() == 'cw':
        args.attack_save_path = f"{args.attack_name}_{args.cw_kappa}_{args.cw_lr}"
    elif args.attack_name.lower() == 'pgd':
        args.attack_save_path = f"{args.attack_name}_{args.eps}"
    else:
        args.attack_save_path = f"{args.attack_name}"

    if args.target_attack.lower() == 'cw':
        args.target_attack_save_path = f"{args.target_attack}_{args.cw_kappa}_{args.cw_lr}"
    elif args.target_attack.lower() == 'pgd':
        args.target_attack_save_path = f"{args.target_attack}_{args.eps}"
    else:
        args.target_attack_save_path = f"{args.target_attack}"

    return args
