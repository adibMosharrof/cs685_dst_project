import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
import json
import dirtyjson
import numpy as np
import glob
import random
import csv
from tqdm import tqdm

def run():
    args = get_args()
    dstc_root = Path('data/dstc8-schema-guided-dialogue')
    processed_path = Path('processed_data')

    steps = ['train', 'dev', 'test']
    for step in tqdm(steps):
        step_dir = Path(processed_path/step)
        step_dir.mkdir(parents=True, exist_ok=True)
        intent_path = dstc_root / step / 'schema.json'
        prepped_intents_path = step_dir / f'intents_{step}.txt'
        contrastive_intent_path = step_dir /f'contrastive_intents_data_{step}.csv'
        prepare_intents(intent_path, prepped_intents_path)
        # prepare_intent_contrastive_data(dstc_root/step, prepped_intents_path, contrastive_intent_path)
        prepare_intents_slots_data(dstc_root/step, prepped_intents_path, contrastive_intent_path)

def prepare_intents(json_path, out_path):
    with open(json_path) as f:
        schemas = json.load(f)
    all_intents = []
    for schema in schemas:
        for intent in schema['intents']:
            all_intents.append(intent['name'])
    unique_intents = np.unique(all_intents)
    with open(out_path, 'w') as f:
        json.dump(unique_intents.tolist(), f)
    a=1


def get_intents_from_dialog(dialogues, intents_data, all_intents):
    for d in dialogues:
            for turn in d['turns']:
                for frames in turn['frames']:
                    for action in frames['actions']:
                            if action['act'] != 'INFORM_INTENT':
                                continue
                            intents = action['canonical_values']
                            for intent in intents:
                                utterance = turn['utterance']
                                pos = [utterance, intent, 1]
                                while True:
                                    neg_intent = random.choice(all_intents)
                                    if neg_intent == intent:
                                        continue
                                    neg = [utterance, neg_intent, 0]
                                    break
                                intents_data.append(pos)
                                intents_data.append(neg)


def prepare_intents_slots_data(path, intents_path, contrastive_intent_path):
    file_paths = glob.glob(str(path / 'dialogues_*'))
    with open(intents_path, 'r') as f:
        all_intents = json.load(f)
    intents_data = []
    for path in tqdm(file_paths):
        with open(path, 'r') as f:
            # dialogues = json.load(f)
            dialogues = dirtyjson.load(f)
        get_intents_from_dialog(dialogues, intents_data, all_intents)
        # for d in dialogues:
        #     for turn in d['turns']:
        #         for frames in turn['frames']:
        #             for action in frames['actions']:
        #                     if action['act'] != 'INFORM_INTENT':
        #                         continue
        #                     intents = action['canonical_values']
        #                     for intent in intents:
        #                         utterance = turn['utterance']
        #                         pos = [utterance, intent, 1]
        #                         while True:
        #                             neg_intent = random.choice(all_intents)
        #                             if neg_intent == intent:
        #                                 continue
        #                             neg = [utterance, neg_intent, 0]
        #                             break
        #                         data.append(pos)
        #                         data.append(neg)

    with open(contrastive_intent_path, 'w', encoding='UTF8', newline="") as f:
        writer = csv.writer(f)
        header = ['utterance', 'intent', 'label']
        writer.writerow(header)
        writer.writerows(intents_data)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-ni", "--num_items", type=int, default=None, help="Data size"
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=2, help="Batch Size"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=2, help="Number of workers"
    )
    parser.add_argument(
        "-si", "--start_index", type=int, default=0, help="Starting index"
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="Gpu number"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Max Epochs"
    )
    parser.add_argument(
        "-ls", "--log_step", type=int, default=1000, help="Log Step"
    )
    parser.add_argument(
        "-ds", "--dataset", type=str, default="out/object_detection_0_100.csv", help="CSV Dataset Path"
    )
    return parser.parse_args()

if __name__ == "__main__":
    run()