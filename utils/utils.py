from __future__ import division, absolute_import, print_function

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Normalize(nn.Module):
    def __init__(self, model, dataset):
        super(Normalize, self).__init__()

        if dataset.lower() == "mnist":
            m, s = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
        elif dataset.lower() == "cifar10":
            m, s = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        elif dataset.lower() == "cifar100":
            m, s = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        elif dataset.lower() == "fmnist":
            m, s = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
        elif dataset.lower() == "svhn":
            m, s = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]

        self.register_buffer('mean', torch.Tensor(m))
        self.register_buffer('std', torch.Tensor(s))

        self.model = model

    def forward(self, inputs):
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        x = (inputs - mean) / std

        return self.model(x)

def network_initialization(args):
    net = __import__('baselineCNN').__dict__[args.model.lower()](args.num_class)
    net = Normalize(net, args.dataset)

    return net.to(args.device)

def get_dataloader(args):
    transformer = _get_transformer(args)
    trn_loader, dev_loader, tst_loader = _get_loader(args, transformer)

    return trn_loader, dev_loader, tst_loader

def _get_loader(args, transformer):
    root = args.data_root_path
    data_path = os.path.join(root, args.dataset.lower())
    data_name = 'FashionMNIST' if args.dataset.lower() == 'fmnist' else args.dataset.upper()
    dataset = getattr(torchvision.datasets, data_name)

    # set transforms
    trn_transform, tst_transform = transformer
    # call dataset
    # normal training set
    if data_name == 'SVHN':
        trainset = dataset(root=data_path, download=True, split='train', transform=trn_transform)
        trainset, devset = torch.utils.data.random_split(trainset, [int(len(trainset) * 0.7)+1, int(len(trainset) * 0.3)])
        tstset = dataset(root=data_path, download=True, split='test', transform=tst_transform)
    else:
        trainset = dataset(root=data_path, download=True, train=True, transform=trn_transform)
        trainset, devset = torch.utils.data.random_split(trainset, [int(len(trainset) * 0.7), int(len(trainset) * 0.3)])
        # validtaion, testing set
        tstset = dataset(root=data_path, download=True, train=False, transform=tst_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        drop_last=True
    )
    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        drop_last=True
    )
    tstloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )

    return trainloader, devloader, tstloader

def get_optim(model, lr):
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
        # model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    return optimizer, scheduler

def _get_transformer(args):
    # with data augmentation
    if 'mnist' in args.dataset:
        trn_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        trn_transformer = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.RandomResizedCrop(32),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    # transformer for testing (validation)
    dev_transformer = transforms.Compose([transforms.ToTensor()])

    return trn_transformer, dev_transformer

