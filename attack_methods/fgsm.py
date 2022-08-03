"""
FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class FGSM(Attack):
    def __init__(self, model, args):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = args.eps
        self.criterion_CE = nn.CrossEntropyLoss()

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach()

        imgs.requires_grad = True

        outputs = self.model(imgs)

        loss = self.criterion_CE(outputs, labels)

        grad = torch.autograd.grad(
            loss, imgs, retain_graph=False, create_graph=False)[0]
        # print(torch.count_nonzero(grad.sign()))
        adv_imgs = imgs+(self.eps*grad.sign())
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels