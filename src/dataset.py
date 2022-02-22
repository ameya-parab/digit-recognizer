import os

import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join("..", "data")


class Digits(Dataset):
    def __init__(self, split: str):
        self._load_data(split)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.images[idx], self.labels[idx]

    def _load_data(self, split):

        data = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))

        if split == "train":
            self.labels = torch.from_numpy(data.label.values).type(torch.LongTensor)
        else:
            self.labels = torch.zeros(data.index.max() + 1).type(torch.LongTensor)

        self.images = torch.from_numpy(
            data[[c for c in data.columns if "pixel" in c]]
            .to_numpy()
            .reshape((-1, 28, 28))
        ).type(torch.FloatTensor)
