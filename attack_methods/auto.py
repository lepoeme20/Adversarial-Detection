import time

from attacks import Attack
from .multiattack import MultiAttack
from .apgd import APGD
from .apgdt import APGDT
from .fab import FAB
from .square import SQUARE


class AUTO(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, args):
        super().__init__("AutoAttack", model)
        norm = 'Linf'
        eps = args.eps
        version = 'standard'
        n_classes = args.num_class
        seed = args.seed
        verbose = False
        self.supported_mode = ['default']

        if version == 'standard':  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=seed, verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, args=args),
                SQUARE(model, args, eps=eps, norm=norm, seed=seed, verbose=verbose, n_queries=5000, n_restarts=1),
            ])

        elif version == 'plus':  # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self._autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=seed, verbose=verbose, loss='ce', n_restarts=5),
                APGD(model, eps=eps, norm=norm, seed=seed, verbose=verbose, loss='dlr', n_restarts=5),
                FAB(model, eps=eps, norm=norm, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=5),
                SQUARE(model, eps=eps, norm=norm, seed=seed, verbose=verbose, n_queries=5000, n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=seed, verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=seed, verbose=verbose, multi_targeted=True, n_classes=n_classes, n_restarts=1),
            ])

        elif version == 'rand':  # ['apgd-ce', 'apgd-dlr']
            self._autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=seed, verbose=verbose, loss='ce', eot_iter=20, n_restarts=1),
                APGD(model, eps=eps, norm=norm, seed=seed, verbose=verbose, loss='dlr', eot_iter=20, n_restarts=1),
            ])

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images, labels = self._autoattack(images, labels)

        return adv_images, labels


    def get_seed(self):
        return time.time() if self.seed is None else self.seed