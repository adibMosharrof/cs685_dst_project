import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from dataclass_csv import DataclassReader
from sentence_transformers import InputExample
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import utils
from my_dataclasses import IntentCsvData, SlotValueCsvData, Steps


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
            collate_fn=getattr(self, "my_collate", None),
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["dev"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=getattr(self, "my_collate", None),
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=getattr(self, "my_collate", None),
        )

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


class IntentDataset(Dataset):
    def __init__(
        self, data: list[IntentCsvData], tokenizer=None, max_token_len=128
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def _convert_str(self, text: str):

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

    def __getitem__(self, index: int):
        item: IntentCsvData = self.data[index]
        return {"x": self._convert_str(item.utterance), "y": item.label}

    def __len__(self):
        return len(self.data)


class ContrastiveIntentDataModule(IntentDataModule):
    def __init__(
        self,
        batch_size=300,
        num_workers=8,
        data_path="processed_data",
        model_name="roberta",
        max_token_len=128,
    ) -> None:
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
                intent_data,
                self.tokenizer,
                self.max_token_len,
            )


class ContrastiveIntentDataset(Dataset):
    def __init__(
        self,
        data: list[IntentCsvData],
        tokenizer=None,
        max_token_len=128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __getitem__(self, index: int):
        item: IntentCsvData = self.data[index]
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


class SlotNameDataModule(IntentDataModule):
    def __init__(
        self,
        batch_size=300,
        num_workers=8,
        data_path="processed_data",
        model_name="roberta",
        max_token_len=128,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.datasets: Dict[str, Dataset] = {}
        for step in Steps.keys():
            csv_path = self.data_path / step / f"slot_value_data_{step}.csv"
            slot_value_data = utils.read_csv_dataclass(csv_path, SlotValueCsvData)
            self.datasets[step] = SlotNameDataset(
                slot_value_data,
                self.tokenizer,
                self.max_token_len,
                step,
                self.data_path,
            )


class SlotNameDataset(Dataset):
    def __init__(
        self,
        data: list[SlotValueCsvData],
        tokenizer=None,
        max_token_len=128,
        step="train",
        data_root: Path = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        slots_path = data_root / step / f"slot_names_{step}.txt"
        slots = utils.read_json(slots_path)
        self.slot_names_dict = {slot: i for i, slot in enumerate(slots)}

    def __getitem__(self, index: int):
        item: SlotValueCsvData = self.data[index]

        return [
            item.utterance,
            item.slot,
            self.slot_names_dict[item.slot],
        ]

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
