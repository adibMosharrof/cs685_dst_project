from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from dataclass_csv import DataclassReader
from sentence_transformers import InputExample
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import utils
from my_dataclasses import IntentCsvData, Steps
import importlib


class IntentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=300,
        num_workers=0,
        data_path="processed_data",
        model_name="roberta-base",
        max_token_len=128,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.datasets: Dict[str, Dataset] = {}
        for step in Steps.keys():
            csv_path = self.data_path / step / f"contrastive_intents_data_{step}.csv"
            intent_data = utils.read_csv_dataclass(csv_path, IntentCsvData)
            self.datasets[step] = IntentDataset(
                intent_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["dev"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ContrastiveIntentDataModule(IntentDataModule):
    def __init__(
        self,
        batch_size=300,
        num_workers=8,
        data_path="processed_data",
        model_name="roberta",
        max_token_len=128,
    ) -> None:
        # super().super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.datasets: Dict[str, Dataset] = {}
        for step in Steps.keys():
            csv_path = self.data_path / step / f"contrastive_intents_data_{step}.csv"
            intent_data = utils.read_csv_dataclass(csv_path, IntentCsvData)
            self.datasets[step] = ContrastiveIntentDataset(
                intent_data, self.tokenizer, self.max_token_len
            )


class ContrastiveIntentDataset(Dataset):
    def __init__(self, data: list[IntentCsvData], tokenizer=None, max_token_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __getitem__(self, index: int):
        item: IntentCsvData = self.data[index]
        # return torch.tensor(InputExample(texts=[item.utterance, item.intent]), dtype=obj)
        return self.tokenize(item.utterance), self.tokenize(item.intent), item.intent

    def tokenize(self, text: str):
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens.input_ids.flatten(),
            "attention_mask": tokens.attention_mask.flatten(),
        }

    def __len__(self):
        return len(self.data)


class IntentDataset(Dataset):
    def __init__(
        self, data: list[IntentCsvData], tokenizer=None, max_token_len=128
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    # def _convert_str(self, text: str, label: str):
    def _convert_str(self, text: str):
        # return self.tokenizer(
        #     text,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.max_token_len,
        #     return_tensors="pt",
        # )
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens.input_ids.flatten(),
            "attention_mask": tokens.attention_mask.flatten(),
            # "labels": label,
        }
        # return tokens

    def __getitem__(self, index: int):
        item: IntentCsvData = self.data[index]
        # return self._convert_str(item.utterance, item.label)
        return {"x": self._convert_str(item.utterance), "y": item.label}
        # utt_enc = self._convert_str(item.utterance, item.label)
        # intent_enc = self._convert_str(item.intent)
        # return {"utterance": utt_enc, "intent": intent_enc, "label": item.label}
        # return IntentCsvData(utterance=utt_enc, intent=intent_enc, label=item.label)
        # return InputExample(texts=[item.utterance, item.intent], label=item.label)

        # return np.array(item.utterance), torch.tensor(item.label)

    def __len__(self):
        return len(self.data)
