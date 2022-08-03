"""
PGD
in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class PGD(Attack):
    def __init__(self, model, args):
        super(PGD, self).__init__("PGD", model)
        self.eps = args.eps
        self.alpha = self.eps/4
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion_CE = nn.CrossEntropyLoss()

    def forward(self, imgs, labels):
        adv_imgs = imgs.clone().detach()

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        for _ in range(self.n_iters):
            adv_imgs.requires_grad = True
            outputs = self.model(adv_imgs)

            loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(
                loss, adv_imgs, retain_graph=False, create_graph=False
            )[0]

            adv_imgs = adv_imgs.detach() + self.alpha*grad.sign()
            eta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + eta, 0, 1).detach()

        return adv_imgs, labels