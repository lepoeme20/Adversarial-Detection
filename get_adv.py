# -*- coding: utf-8 -*-
import os
import torch
from utils.utils import (
    network_initialization,
    get_dataloader,
    set_seed
)
import config
from attack_methods import df, pgd, fgsm, bim, cw, fab, auto, pgd2, square


class GetADV:
    def __init__(self, args):
        set_seed(args.seed)
        self.model = network_initialization(args)
        model_path = os.path.join(args.save_path, args.model, f"{args.dataset}.pt")
        pretrained_dict = torch.load(model_path)["model_state_dict"]
        # pretrained_dict = {f'model.{k}': v for k, v in pretrained_dict.items()}
        self.model.load_state_dict(
            pretrained_dict
        )
        self.model.eval()

    def attack(self, tstloader):
        attack_module = globals()[args.attack_name.lower()]
        attack_func = getattr(attack_module, args.attack_name)
        attacker = attack_func(self.model, args)
        save_path = os.path.join("adv_examples", args.model, args.dataset.lower())
        attacker.inference(
            args,
            data_loader=tstloader,
            save_path=save_path,
            file_name=f"{args.attack_save_path}.pt"
        )

    def get_adv(self):
        args.batch_size = args.test_batch_size
        _, _, tst_loader = get_dataloader(args)
        self.attack(tst_loader)

if __name__ == "__main__":
    args = config.get_config()
    tester = GetADV(args)

    tester.get_adv()
