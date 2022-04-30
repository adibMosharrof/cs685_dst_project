import importlib
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as metrics
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sentence_transformers.cross_encoder import CrossEncoder
from torch import nn, optim
from transformers import AutoModel

import utils
from my_dataclasses import Steps


class IntentModel(pl.LightningModule):
    def __init__(self, n_classes=5, model_name="roberta-base"):
        super().__init__()
        self.n_classes = n_classes
        # sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True)

        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = metrics.MetricCollection(
            {
                "top-01": metrics.Accuracy(),
                # "top-01": metrics.Accuracy(top_k=1),
                # "top-02": metrics.Accuracy(top_k=2),
                # "top-03": metrics.Accuracy(top_k=3),
                # "top-05": metrics.Accuracy(top_k=5),
                # "top-10": metrics.Accuracy(top_k=10),
            }
        )
        self.val_metrics = metrics.MetricCollection(
            {
                "top-01": metrics.Accuracy(),
                # "top-01": metrics.Accuracy(top_k=1),
                # "top-02": metrics.Accuracy(top_k=2),
                # "top-03": metrics.Accuracy(top_k=3),
                # "top-05": metrics.Accuracy(top_k=5),
                # "top-10": metrics.Accuracy(top_k=10),
            }
        )
        self.test_metrics = metrics.MetricCollection(
            {
                "top-01": metrics.Accuracy(),
                # "top-01": metrics.Accuracy(top_k=1),
                # "top-02": metrics.Accuracy(top_k=2),
                # "top-03": metrics.Accuracy(top_k=3),
                # "top-05": metrics.Accuracy(top_k=5),
                # "top-10": metrics.Accuracy(top_k=10),
            }
        )

    def forward(self, data):
        # return self.model(**data)
        # return self.model(data["input_ids"])
        out = self.bert(data["input_ids"], data["attention_mask"])
        x = torch.mean(out.last_hidden_state, 1)
        x = F.relu(x)
        # x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="train", metric=self.train_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="val", metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step="test", metric=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        data, labels = batch.values()
        logits = self(data)
        # labels = batch["labels"]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = metric(preds, labels)
        # accuracy = metric(logits, labels).detach()
        self.log(f"{step}/acc", accuracy, prog_bar=True)
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)


class BaseModel(pl.LightningModule):
    def __init__(
        self, model_name="distilbert-base-uncased", loss_func: nn.Module = None
    ):
        super().__init__()

        self.model = SentenceTransformer(model_name)

        self.criterion = loss_func(model=self.model)
        self.train_metrics = metrics.MetricCollection(
            {
                "train_top-01": metrics.Accuracy(),
                "train_top-01": metrics.Accuracy(top_k=1),
                "train_top-02": metrics.Accuracy(top_k=2),
                "train_top-03": metrics.Accuracy(top_k=3),
                "train_top-05": metrics.Accuracy(top_k=5),
            }
        )
        self.val_metrics = metrics.MetricCollection(
            {
                "val_top-01": metrics.Accuracy(),
                "val_top-01": metrics.Accuracy(top_k=1),
                "val_top-02": metrics.Accuracy(top_k=2),
                "val_top-03": metrics.Accuracy(top_k=3),
                "val_top-05": metrics.Accuracy(top_k=5),
            }
        )
        self.test_metrics = metrics.MetricCollection(
            {
                "test_top-01": metrics.Accuracy(),
                "test_top-01": metrics.Accuracy(top_k=1),
                "test_top-02": metrics.Accuracy(top_k=2),
                "test_top-03": metrics.Accuracy(top_k=3),
                "test_top-05": metrics.Accuracy(top_k=5),
            }
        )

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.train.name, metric=self.test_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.dev.name, metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.train.name, metric=self.test_metrics)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)


class ContrastiveIntentModel(pl.LightningModule):
    def __init__(
        self, model_name="distilbert-base-uncased", data_root="processed_data"
    ):
        super().__init__()

        self.model = SentenceTransformer(model_name)
        # self.model = CrossEncoder(model_name, num_labels=1)

        self.criterion = losses.MultipleNegativesRankingLoss(model=self.model)
        self.metrics = {}
        self.train_metrics = metrics.MetricCollection(
            {
                "train_top-01": metrics.Accuracy(),
                "train_top-01": metrics.Accuracy(top_k=1),
                "train_top-02": metrics.Accuracy(top_k=2),
                "train_top-03": metrics.Accuracy(top_k=3),
                "train_top-05": metrics.Accuracy(top_k=5),
            }
        )
        self.val_metrics = metrics.MetricCollection(
            {
                "val_top-01": metrics.Accuracy(),
                "val_top-01": metrics.Accuracy(top_k=1),
                "val_top-02": metrics.Accuracy(top_k=2),
                "val_top-03": metrics.Accuracy(top_k=3),
                "val_top-05": metrics.Accuracy(top_k=5),
            }
        )
        self.test_metrics = metrics.MetricCollection(
            {
                "test_top-01": metrics.Accuracy(),
                "test_top-01": metrics.Accuracy(top_k=1),
                "test_top-02": metrics.Accuracy(top_k=2),
                "test_top-03": metrics.Accuracy(top_k=3),
                "test_top-05": metrics.Accuracy(top_k=5),
            }
        )

        self.all_intents_dict = {}
        self.all_intents = {}
        for step in ["dev", "test", "train"]:
            path = Path(data_root) / step / f"intents_{step}.txt"
            intents = utils.read_json(str(path))
            self.all_intents_dict[step] = {k: i for i, k in enumerate(intents)}
            self.all_intents[step] = list(self.all_intents_dict[step].keys())

    def training_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.train.name, metric=self.test_metrics)

    def validation_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.dev.name, metric=self.val_metrics)

    def test_step(self, batch, batch_idx=None):
        return self._shared_step(batch, step=Steps.train.name, metric=self.test_metrics)

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        utterances, intents, intents_str = batch
        loss = self.criterion([utterances, intents], 1)
        intents_emb = self.model.encode(self.all_intents[step], convert_to_tensor=True)
        similarity = util.cos_sim(utterances["sentence_embedding"], intents_emb)
        labels = torch.tensor(
            [self.all_intents_dict[step][intent] for intent in intents_str],
            device=self.model.device,
        )
        self.log_dict(metric(similarity, labels), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)


class SlotNameModel(BaseModel):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        data_root="processed_data",
        loss_func_name: str = "OnlineContrastiveLoss",
    ):
        # super().__init__(model_name, losses.OnlineContrastiveLoss)
        loss_func = getattr(
            importlib.import_module("sentence_transformers.losses"), loss_func_name
        )
        super().__init__(model_name, loss_func)
        self.model = SentenceTransformer(model_name)
        self.slot_names_dict = {}
        self.slot_names = {}
        for step in ["dev", "test", "train"]:
            path = Path(data_root) / step / f"slot_names_{step}.txt"
            slots = utils.read_json(str(path))
            self.slot_names_dict[step] = {slot: i for i, slot in enumerate(slots)}
            self.slot_names[step] = list(self.slot_names_dict[step].keys())

    # def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
    #     pos, neg = batch
    #     utterances = {
    #         "input_ids": torch.concat((pos[0]["input_ids"], neg[0]["input_ids"])),
    #         "attention_mask": torch.concat(
    #             (pos[0]["attention_mask"], neg[0]["attention_mask"])
    #         ),
    #     }

    #     slots = {
    #         "input_ids": torch.concat((pos[1]["input_ids"], neg[1]["input_ids"])),
    #         "attention_mask": torch.concat(
    #             (pos[1]["attention_mask"], neg[1]["attention_mask"])
    #         ),
    #     }
    #     labels = torch.concat((pos[2], neg[2]))
    #     slot_names_id = torch.concat((pos[3], neg[3]))

    #     loss = self.criterion([utterances, slots], labels)
    #     slot_name_emb = self.model.encode(self.slot_names[step], convert_to_tensor=True)
    #     similarity = util.cos_sim(utterances["sentence_embedding"], slot_name_emb)
    #     self.log_dict(metric(similarity, slot_names_id), prog_bar=True)
    #     return loss

    def _shared_step(self, batch, batch_idx=None, step="train", metric=None):
        utterances, slots, slot_names_id = batch
        slot_names_id = slot_names_id[0]
        # utterances = {
        #     "input_ids": torch.concat((pos[0]["input_ids"], neg[0]["input_ids"])),
        #     "attention_mask": torch.concat(
        #         (pos[0]["attention_mask"], neg[0]["attention_mask"])
        #     ),
        # }

        # slots = {
        #     "input_ids": torch.concat((pos[1]["input_ids"], neg[1]["input_ids"])),
        #     "attention_mask": torch.concat(
        #         (pos[1]["attention_mask"], neg[1]["attention_mask"])
        #     ),
        # }
        # labels = torch.concat((pos[2], neg[2]))
        # slot_names_id = torch.concat((pos[3], neg[3]))

        loss = self.criterion([utterances, slots], 1)
        slot_name_emb = self.model.encode(self.slot_names[step], convert_to_tensor=True)
        similarity = util.cos_sim(utterances["sentence_embedding"], slot_name_emb)
        self.log_dict(metric(similarity, slot_names_id), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-5)
