import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

from deadbeats import DEADBEATS


class ExperimentMixin(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.freezed_modules = []
        self.used_device = None
        self.example_input_array = None


    def forward(self, batch: DefinitionBatch): pass

    def on_train_start(self):
        print("start training!🎉")
        self.unfreeze()
        freeze_modules(self.freezed_modules)
        print("-"*20)
        print("training parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
        print("-"*20)
        self.used_device = self.device


    def on_train_end(self):
        print("👍End training!")


    def on_train_epoch_start(self):
        self.unfreeze()
        freeze_modules(self.freezed_modules)


    def training_step(self, batch, batch_idx):
        loss = self(batch)
        return {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': {'training_loss': loss}}


    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return {'val_loss': loss, 'progress_bar': {'validation_loss': loss}, 'log': {'validation_loss': loss}}


    def test_step(self, batch, batch_idx):
        loss = self(batch)
        return {'test_loss': loss, 'progress_bar': {'test_loss': loss}, 'log': {'test_loss': loss}}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        DEADBEATS.ping(val_loss = avg_loss, current_epoch = self.current_epoch)
        return {'val_loss': avg_loss}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}


    @DEADBEATS.wrap
    def fit(self, trainer):
        print("🩺 sanity checking now...")
        with torch.no_grad():
            self.evaluate(sanity_check=True)
        print("👨‍⚕️sanity check complete!👩‍⚕️")
        trainer.fit(self, self.dm)


    def save(self, seve_path): pass
    def load(self, load_path): pass
