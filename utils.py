import csv
import json
from pathlib import Path

import dirtyjson
import torch
from dataclass_csv import DataclassReader


def write_csv(headers: list[str], data, file_name: str):
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


def write_json(data: list[any], path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def read_json(path: str):
    with open(path, "r") as f:
        data = dirtyjson.load(f)
    return data


def read_csv(path: str):
    fields = []
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        fields = next(reader)
        for r in reader:
            rows.append(r)
    return rows


def read_csv_dataclass(path: str, d_class):
    with open(path) as f:
        reader = DataclassReader(f, d_class)
        return [r for r in reader]


def get_num_items(num, max_value):
    if num == None:
        return max_value
    return num


def read_lines_in_file(path: Path) -> list[any]:
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    return lines


def collate_fn(batch):
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
