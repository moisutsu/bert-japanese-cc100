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
        data_path,
        batch_size = 32,
        learning_rate=1e-5,
        pretrained_model_name = PRETRAINED_JAPANESE_BERT,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.pretrained_model_name = pretrained_model_name

        self.model = BertForPreTraining.from_pretrained(pretrained_model_name, return_dict=True)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
        self.dm = BertJapaneseDataModule(data_path=data_path, tokenizer=self.tokenizer)

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def forward(self, batch):
        input_ids, pred_indexes, true_word_ids, token_type_ids, attention_mask, sop_label = batch

        outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pred = outputs.prediction_logits[pred_indexes]
        sop = outputs.seq_relationship_logits

        pred_loss = self.criterion(pred, true_word_ids)
        sop_loss = self.criterion(sop, sop_label)
        loss = pred_loss + sop_loss
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