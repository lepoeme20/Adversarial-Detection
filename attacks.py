"""Adversarial attack class
"""
import os
import torch

class Attack(object):
    """Base class for attacks

    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, attack_type, model):
        self.attack_name = attack_type
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def forward(self, *args):
        """Call adversarial examples
        Should be overridden by all attack classes
        """
        raise NotImplementedError

    def inference(self, args, save_path, file_name, data_loader):
        """[summary]

        Arguments:
            save_path {[type]} -- [description]
            data_loader {[type]} -- [description]
        """
        adv_list = []
        label_list = []

        correct = 0
        accumulated_num = 0.
        total_num = len(data_loader)

        for step, (imgs, labels) in enumerate(data_loader):
            if imgs.size(1) == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            adv_imgs, labels = self.__call__(imgs, labels)
            adv_list.append(adv_imgs.cpu())
            label_list.append(labels.cpu())

            accumulated_num += labels.size(0)

            outputs = self.model(adv_imgs)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            acc = 100 * correct / accumulated_num

            print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
                (step+1)/total_num*100, acc), end='\r')

        print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
            (step+1)/total_num*100, acc))

        if args.save_adv:
            adversarials = torch.cat(adv_list, 0)
            y = torch.cat(label_list, 0)

            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, file_name)
            torch.save((adversarials, y), save_path)

    def training(self, imgs):
        adv_imgs, labels = self.__call__(imgs, labels)
        return adv_imgs, labels

    def __call__(self, *args):
        adv_examples, labels = self.forward(*args)
        return adv_examples, labels