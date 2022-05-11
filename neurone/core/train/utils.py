import os
import torch
from torch import nn


def weight_init(module):
    """Initialises the model weights"""

    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))
        if module.bias is not None:
            nn.init.uniform_(module.bias, a=-0.1, b=0.1)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.uniform_(module.bias, a=-0.1, b=0.1)


def make_checkpoint(model, optimizer, scheduler, epoch_i, path, overwrite=False):

    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise Exception(
                "The checkpoint exists and overwrite mode is disabled: \n %s \n Aborting."
                % path
            )

    dict_to_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_i": epoch_i,
    }

    if scheduler is not None:
        dict_to_save["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(dict_to_save, path)


def load_checkpoint(model, optimizer, scheduler, path, overwrite=False):

    checkpoint = torch.load("try/checkpoints/epoch_1.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
