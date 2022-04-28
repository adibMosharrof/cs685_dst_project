import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
import json
import numpy as np
import csv
from sentence_transformers import (
    LoggingHandler,
    losses,
    InputExample,
)
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
    CESoftmaxAccuracyEvaluator,
)
import utils
import torchmetrics as metrics
import mpu.ml
from tqdm import tqdm


def test_intent(data_root):
    k = 4
    test_metrics = metrics.MetricCollection(
        {
            "top-01": metrics.Accuracy(top_k=1),
            "top-02": metrics.Accuracy(top_k=2),
            "top-03": metrics.Accuracy(top_k=3),
            "top-05": metrics.Accuracy(top_k=5),
            "top-10": metrics.Accuracy(top_k=10),
        }
    )
    model_out_path = "out/model_checkpoint"
    model = CrossEncoder(model_out_path)
    test_path = data_root / "test"
    test_data_path = test_path / "contrastive_intents_data_test.csv"
    intents_path = test_path / "intents_test.txt"
    all_intents = utils.read_json(intents_path)
    intents_dict = {k: v for v, k in enumerate(all_intents)}
    test_data = get_data(test_data_path)

    for data in tqdm(test_data):
        text, label = data.texts
        test_inputs = [[text, intent] for intent in all_intents]
        logits = torch.tensor(np.array([model.predict(test_inputs)]))
        target_labels = torch.tensor(np.array([intents_dict[label]]))
        acc = test_metrics(logits, target_labels)
    acc = test_metrics.compute()
    print(f"Accuracy on test set {acc}")


def train_contrastive_intent(data_root):
    model_out_path = Path("out/model_checkpoint")
    model_out_path.mkdir(parents=True, exist_ok=True)
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    model = CrossEncoder("distilroberta-base", num_labels=1)
    train_data_path = data_root / "train" / "contrastive_intents_data_train.csv"
    dev_data_path = data_root / "dev" / "contrastive_intents_data_dev.csv"
    test_data_path = data_root / "test" / "contrastive_intents_data_test.csv"
    train_data = get_data(train_data_path)
    train_dl = DataLoader(train_data, shuffle=True, batch_size=300)
    dev_data = get_data(dev_data_path)
    # test_data = get_data(test_data_path)
    # train_loss = losses.ContrastiveLoss(model=model)
    # evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_data)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_data)
    model.fit(
        train_dataloader=train_dl,
        show_progress_bar=True,
        epochs=70,
        evaluator=evaluator,
        output_path=str(model_out_path),
    )
    # model = SentenceTransformer(model_out_path)
    # test_evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_data)
    # test_evaluator(model, output_path=str(model_out_path))
    a = 1


def get_data(data_path):
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        data = [InputExample(texts=[r[0], r[1]], label=int(r[2])) for r in reader]
    return data
    # return DataLoader(data, shuffle=True, batch_size=2)


if __name__ == "__main__":
    data_root = Path("processed_data")
    # train_contrastive_intent(data_root)
    test_intent(data_root)
