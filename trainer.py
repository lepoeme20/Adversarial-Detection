import os
import torch
import torch.nn as nn
from utils.utils import network_initialization, get_dataloader, get_optim, set_seed
from tqdm import tqdm
import config

class Trainer():
    def __init__(self,args):
        set_seed(args.seed)
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization

        self.save_path = os.path.join(args.save_path, args.model)
        os.makedirs(self.save_path, exist_ok=True)

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()

    def training(self):
        model_path = os.path.join(self.save_path, f"{args.dataset}.pt")

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(self.model, args.lr)

        best_loss = 1000
        current_step = 0
        dev_step = 0

        trn_loss_log = tqdm(total=0, position=2, bar_format='{desc}')
        dev_loss_log = tqdm(total=0, position=4, bar_format='{desc}')
        best_epoch_log = tqdm(total=0, position=5, bar_format='{desc}')
        outer = tqdm(total=args.epochs, desc="Epoch", position=0, leave=False)

        # Train target classifier
        for epoch in range(args.epochs):
            _dev_loss = 0.0
            train = tqdm(total=len(self.train_loader), desc="[TRN] Step", position=1, leave=False)
            dev = tqdm(total=len(self.dev_loader), desc="[DEV] Step", position=3, leave=False)
            for _, (inputs, labels) in enumerate(self.train_loader, 0):
                self.model.train()
                current_step += 1
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)

                inputs, labels = inputs.to(args.device), labels.to(args.device)

                # Compute loss
                logit = self.model(inputs)
                loss = self.criterion_CE(logit, labels)

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Loss: {loss.item():.4f}"
                )
                train.update(1)

            for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
                self.model.eval()
                dev_step += 1
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(args.device), labels.to(args.device)

                with torch.no_grad():
                    logit = self.model(inputs)
                    # Compute loss
                    loss = self.criterion_CE(logit, labels)

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss/(idx+1)

                    dev_loss_log.set_description_str(
                        f"[DEV] Loss: {dev_loss.item():.4f}"
                    )
                    dev.update(1)

            scheduler.step(dev_loss)
            outer.update(1)

            if dev_loss < best_loss:
                best_epoch_log.set_description_str(
                    f"Best Epoch: {epoch} / {args.epochs} | Best Loss: {dev_loss}"
                )
                best_loss = dev_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "trained_epoch": epoch,
                    },
                    model_path
                )

#TODO: 
# 1. resnet + cifar10, cifar100, svhn
# 2. vgg + cifar10, cifar100, svhn

if __name__ == "__main__":
    args = config.get_config()
    trainer = Trainer(args)
    args.epochs=200
    args.batch_size=512
    trainer.training()