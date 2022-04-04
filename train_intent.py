
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
import json
import numpy as np
import csv
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def train_contrastive_intent(data_root):
    model_out_path = 'out/model_checkpoint'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_data_path = data_root / "train" / "contrastive_intents_data_train.csv"
    dev_data_path = data_root / "dev" / "contrastive_intents_data_dev.csv"
    test_data_path = data_root / "test" / "contrastive_intents_data_test.csv"
    train_data = get_data(train_data_path)
    train_dl = DataLoader(train_data, shuffle=True, batch_size=256)
    dev_data = get_data(dev_data_path)
    test_data = get_data(test_data_path)
    train_loss = losses.ContrastiveLoss(model=model)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data)
    model.fit(
        [(train_dl, train_loss)], 
        show_progress_bar=True, 
        epochs=10,
        evaluator=evaluator,
        output_path = model_out_path
        )
    model = SentenceTransformer(model_out_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data)
    test_evaluator(model, output_path=model_out_path)
    a=1



def get_data(data_path):
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        data = [InputExample(texts=[r[0], r[1]], label=int(r[2])) for r in reader]
    return data
    # return DataLoader(data, shuffle=True, batch_size=2)

if __name__ == "__main__":
    data_root = Path('processed_data')
    train_contrastive_intent(data_root)