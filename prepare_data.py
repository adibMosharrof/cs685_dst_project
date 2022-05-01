import torch
from argparse import ArgumentParser
from pathlib import Path
import json
import dirtyjson
import numpy as np
import glob
import random
import csv
from tqdm import tqdm
import utils


class PrepareData:
    def run(self):
        args = self.get_args()
        dstc_root = Path("data/dstc8-schema-guided-dialogue")
        processed_path = Path("processed_data")

        steps = ["train", "dev", "test"]
        for step in tqdm(steps):
            step_dir = Path(processed_path / step)
            step_dir.mkdir(parents=True, exist_ok=True)
            schemas_path = dstc_root / step / "schema.json"
            prepped_intents_path = step_dir / f"intents_{step}.txt"
            prepped_slot_names_path = step_dir / f"slot_names_{step}.txt"
            contrastive_intent_path = step_dir / f"contrastive_intents_data_{step}.csv"
            slot_value_path = step_dir / f"slot_value_data_{step}.csv"
            slot_labels_path = step_dir / f"slot_labels_{step}.txt"
            slot_classification_out_path = step_dir / f"slot_classification_{step}.txt"
            schemas = self._get_schemas(schemas_path)
            slot_labels = self.prepare_slot_labels(schemas, slot_labels_path)
            # self.prepare_slot_classification(
            #     step_dir, slot_classification_out_path, slot_labels
            # )
            self.prepare_intents(schemas, prepped_intents_path)
            self.prepare_intents(schemas, prepped_slot_names_path, field_name="slots")
            # self.prepare_intents_slots_data(
            #     dstc_root / step,
            #     prepped_intents_path,
            #     contrastive_intent_path,
            #     step=step,
            # )
            # self.prepare_slot_values_data(dstc_root / step, slot_value_path)

    def prepare_slot_classification(self, step_dir, out_path, slot_labels):
        """
        The slot labels are converted from string to int for classification
        Returns utterance, class id pair
        """
        slot_labels_dict = {
            k: v for k, v in zip(slot_labels, list(range(len(slot_labels))))
        }

        a = 1

    def prepare_intents(
        self, schemas, out_path, field_name="intents", item_name="name"
    ):
        all_intents = ["NONE"]
        for schema in tqdm(schemas):
            for intent in schema[field_name]:
                all_intents.append(intent[item_name])
        unique_intents = np.unique(all_intents)
        utils.write_json(unique_intents.tolist(), out_path)

    def prepare_slot_labels(self, schemas, out_path):
        all_slot_labels = []
        for schema in schemas:
            for slot in schema["slots"]:
                val = f"{slot['name']}"
                all_slot_labels.append(val)

            utils.write_json(all_slot_labels, out_path)
            return all_slot_labels

    def get_intents_from_dialog(
        self, dialogues, intents_data, all_intents, step="train"
    ):
        for d in dialogues:
            for turn in d["turns"]:
                if turn["speaker"] == "SYSTEM":
                    continue
                for frames in turn["frames"]:
                    # for action in frames["actions"]:
                    # if action["act"] != "INFORM_INTENT":
                    #     continue
                    utterance = turn["utterance"]
                    intent = frames["state"]["active_intent"]
                    pos = [utterance, intent, 1]
                    intents_data.append(pos)
                    if step in ["test", "dev", "train"]:
                        continue
                    while True:
                        neg_intent = random.choice(all_intents)
                        if neg_intent == intent:
                            continue
                        neg = [utterance, neg_intent, 0]
                        break
                    intents_data.append(neg)

    def prepare_intents_slots_data(
        self, dstc_step_path, intents_path, contrastive_intent_path, step="train"
    ):
        dialog_json_paths = self.get_dialogues_file_paths(dstc_step_path)
        all_intents = utils.read_json(intents_path)
        intents_data = []
        for path in tqdm(dialog_json_paths):
            dialogues = utils.read_json(path)
            self.get_intents_from_dialog(
                dialogues, intents_data, all_intents, step=step
            )

        headers = ["utterance", "intent", "label"]
        utils.write_csv(headers, intents_data, contrastive_intent_path)

    def get_slot_values_from_dialog(self, dialogues, step="train"):
        data = []
        for d in dialogues:
            for turn in d["turns"]:
                if turn["speaker"] == "SYSTEM":
                    continue
                for frames in turn["frames"]:
                    utterance = turn["utterance"]
                    slots = frames["slots"]
                    for slot in slots:
                        data.append(
                            [
                                utterance,
                                slot["slot"],
                                slot["start"],
                                slot["exclusive_end"],
                            ]
                        )
        return data

    def prepare_slot_values_data(
        self, dstc_step_path, slot_value_out_path, step="train"
    ):
        dialog_json_paths = self.get_dialogues_file_paths(dstc_step_path)
        for path in tqdm(dialog_json_paths):
            dialogues = utils.read_json(path)
            data = self.get_slot_values_from_dialog(dialogues, step=step)

        headers = ["utterance", "slot", "start_index", "end_index"]
        utils.write_csv(headers, data, slot_value_out_path)

    def _get_schemas(self, path):
        with open(path) as f:
            schemas = json.load(f)
        return schemas

    def get_dialogues_file_paths(self, path):
        file_paths = glob.glob(str(path / "dialogues_*"))
        return file_paths

    def _write_json(self, path, headers, rows):
        with open(path, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def get_args(self):
        parser = ArgumentParser()
        parser.add_argument("-ni", "--num_items", type=int, default=2, help="Data size")
        parser.add_argument(
            "-bs", "--batch_size", type=int, default=2, help="Batch Size"
        )
        parser.add_argument(
            "-w", "--workers", type=int, default=2, help="Number of workers"
        )
        parser.add_argument(
            "-si", "--start_index", type=int, default=0, help="Starting index"
        )

        return parser.parse_args()


if __name__ == "__main__":
    prep_data = PrepareData()
    prep_data.run()
