"""
Iterative FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class BIM(Attack):
    def __init__(self, model, args):
        super(BIM, self).__init__("BIM", model)
        self.eps = args.eps
        self.alpha = 2/255
        self.step = args.bim_step
        self.criterion_CE = nn.CrossEntropyLoss()

    def forward(self, imgs, labels):
        imgs = imgs.clone().detach().to(self.device)
        orig_imgs = imgs.clone().detach().to(self.device)

        for _ in range(self.step):
            imgs.requires_grad = True
            outputs = self.model(imgs)

            loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(
                loss, imgs, retain_graph=False, create_graph=False)[0]

            adv_imgs = imgs + self.alpha*grad.sign()
            # a = max(ori_images-eps, 0)
            a = torch.clamp(orig_imgs - self.eps, min=0)
            # b = max(adv_images, a) = max(adv_images, ori_images-eps, 0)
            b = (adv_imgs >= a).float()*adv_imgs + (adv_imgs < a).float()*a
            # c = min(ori_images+eps, b) = min(ori_images+eps, max(adv_images, ori_images-eps, 0))
            c = (b > orig_imgs+self.eps).float()*(orig_imgs+self.eps) + (b <= orig_imgs + self.eps).float()*b
            # images = max(1, c) = min(1, ori_images+eps, max(adv_images, ori_images-eps, 0))
            imgs = torch.clamp(c, max=1).detach()

        return adv_imgs.detach(), labels