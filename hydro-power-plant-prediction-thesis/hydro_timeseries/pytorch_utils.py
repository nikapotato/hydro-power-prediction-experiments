'''
Pytorch specific utils
Loss fns
- Noisy 15 min data, perhaps MAE?

Early stoppers
'''
import torch
from pytorch_lightning.callbacks import EarlyStopping


def MAELoss(yhat,y):
    return torch.mean(torch.abs(yhat - y))

def ModifiedMAELoss(yhat,y):
    # return torch.mean(torch.abs(yhat - y))
    return torch.mean(torch.abs(yhat[:, :24, :]-y[:, :24, :])) +\
           2 * torch.mean(torch.abs(yhat[:, 24:, :]-y[:, 24:, :]))


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MSELoss(yhat,y):
    return torch.mean((yhat-y)**2)

early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.01,
    mode='min',
)

pl_trainer_kwargs={"callbacks": [early_stopper]}
