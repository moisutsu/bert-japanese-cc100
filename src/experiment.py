import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertForPreTraining, BertJapaneseTokenizer

from .data_module import BertJapaneseDataModule

PRETRAINED_JAPANESE_BERT = "cl-tohoku/bert-base-japanese"


class Experiment(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-5,
        pretrained_model_name = PRETRAINED_JAPANESE_BERT,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.pretrained_model_name = pretrained_model_name

        self.model = BertForPreTraining.from_pretrained(pretrained_model_name)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
        self.dm = BertJapaneseDataModule(tokenizer=self.tokenizer)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def forward(self, batch):
        input_ids, pred_indexes, true_word_ids, token_type_ids, attention_mask, sop_label = batch
        print("input_ids", input_ids, input_ids.size())
        print("pred_indexes", pred_indexes, pred_indexes.size())
        print("true_word_ids", true_word_ids, true_word_ids.size())
        print("token_type_ids", token_type_ids, token_type_ids.size())
        print("attention_mask", attention_mask)
        print("sop_label", sop_label)

        exit(1)

        out = self.model()
        loss = self.criterion(out)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self(batch)
        return {'loss': loss, 'progress_bar': {'training_loss': loss}, 'log': {'training_loss': loss}}


    def save(self, trainer, save_path):
        trainer.save_checkpoint(save_path)


    @classmethod
    def load(cls, save_path, **kwargs):
        return cls.load_from_checkpoint(checkpoint_path=save_path, **kwargs)

    def fit(self, trainer):
        trainer.fit(self, self.dm)