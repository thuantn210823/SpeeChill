from typing import Callable, Optional
from omegaconf import DictConfig

import torch
import lightning as L
from torch.nn.utils.rnn import pad_sequence

class Dataloader(L.LightningDataModule):
    def __init__(self,
                 train_dataset: DictConfig,
                 val_dataset: DictConfig,
                 test_dataset: DictConfig = None,
                 loaders: DictConfig = None):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.loaders = loaders
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           shuffle = True,
                                           collate_fn = self.collate_fn,
                                           **self.hparams.loaders
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           shuffle = False,
                                           collate_fn = self.collate_fn,
                                           **self.hparams.loaders
                                           )

    def test_dataloader(self):
        if self.test_dataset is not None:
            return torch.utils.data.DataLoader(self.test_dataset,
                                               shuffle = False,
                                               collate_fn = self.collate_fn,
                                               **self.hparams.loaders
                                               )
        return None

    def collate_fn(self, batch):
        _batch = {'feats': [],
                  'feat_lengths': [],
                  'targets': [],
                  'tasks': []}
        for sample in batch:
            _batch['feats'].append(sample['feat'].transpose(0, 1))
            _batch['feat_lengths'].append(sample['feat'].shape[-1])
            _batch['tasks'].append(sample['task'])
            _batch['targets'].append(sample['text'])
        _batch['feats'] = pad_sequence(_batch['feats'], batch_first = True)
        _batch['feat_lengths'] = torch.tensor(_batch['feat_lengths'], dtype = torch.long)
        return _batch

class TurnTaking(L.LightningModule):
    def __init__(self,
                 model: Callable,
                 lr: float,
                 optimizer: Callable,
                 scheduler: Callable,
                 ckpt: Optional[DictConfig] = None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        if ckpt is not None:
            self.load_ckpt(ckpt)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch, self.device)
        loss = loss_dict['loss']

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model(batch, self.device)
        loss = loss_dict['loss']

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params = self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer = optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "step",
                    "frequency": 1
                }
            }
        return {"optimizer": optimizer}

    def load_ckpt(self, ckpt):
        ckpt = torch.load(ckpt, weights_only=False, map_location='cpu')
        state_dict = self.state_dict()
        for key in state_dict.keys():
            for key in ckpt['state_dict'].keys():
                state_dict[key] = ckpt['state_dict'][key]
        self.load_state_dict(state_dict)