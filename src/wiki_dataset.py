import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

from tqdm import tqdm
from pathlib import Path

class WikiDataset(Dataset):
    def __init__(
        self,
        transform,
        datapath=Path("wiki_formated.txt"),
    ):
        super().__init__()

        self.transform = transform

        self.dataset = []
        with datapath.open() as f:
            prev = next(f).strip()
            for line in tqdm(f):
                sentence = line.strip()
                if sentence == "":
                    prev = None
                    continue
                if prev is not None:
                    self.dataset.append((prev, sentence))
                prev = sentence


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, key):
        if isinstance(key, int):
            return self.transform(self.dataset[key])
        else:
            raise ValueError("invalid slice")