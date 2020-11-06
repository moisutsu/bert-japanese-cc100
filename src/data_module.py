import torch
import torch.nn as nn
import pytorch_lightning as pl
import os, sys
import random

from collections import defaultdict
from torch.utils.data import DataLoader
from pathlib import Path

from .wiki_dataset import WikiDataset


class BertJapaneseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        data_path,
        batch_size = 32,
        max_sequence_length = 64,
        num_workers = os.cpu_count(),
        dataset_name = "wiki",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = Path(data_path)

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_workers = num_workers
        self.dataset_name = dataset_name


    def setup(self, stage=None):
        if self.dataset_name == "wiki":
            self.dataset = WikiDataset(data_path=self.data_path, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle = True,
        )


    def transform(self, data):
        first, second = data
        reverse = random.random() > 0.5
        if reverse:
            first, second = second, first
        sop_label = torch.tensor([int(reverse)])

        tokenized = self.tokenizer(first, second, max_length=self.max_sequence_length, padding="max_length", return_tensors="pt", truncation=True)
        input_ids, token_type_ids, attention_mask = tokenized.values()
        P = torch.rand(input_ids.size())
        pred_indexes = (P >= 0.85) * (input_ids != self.tokenizer.pad_token_id) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id)

        true_word_ids = input_ids[pred_indexes]

        # first, second, torch.tensor([int(reverse)])
        mask_flag = pred_indexes * (0.85 <= P) * (P < 0.97)
        replace_flag = pred_indexes * (P >= 0.985)

        input_ids[mask_flag] = self.tokenizer.mask_token_id
        input_ids[replace_flag] = torch.randint(low=5, high=self.tokenizer.vocab_size, size=(replace_flag.sum(),))

        return (input_ids, pred_indexes, true_word_ids, token_type_ids, attention_mask, sop_label)


    def collate_fn(self, data_list):
        input_ids, pred_indexes, true_word_ids, token_type_ids, attention_mask, sop_label = list(zip(*data_list))

        return (
            torch.cat(input_ids, dim=0),
            torch.cat(pred_indexes, dim=0),
            torch.cat(true_word_ids),
            torch.cat(token_type_ids, dim=0),
            torch.cat(attention_mask, dim=0),
            torch.cat(sop_label),
        )