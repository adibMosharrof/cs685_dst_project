from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    MODEL_REGISTRY,
    LightningArgumentParser,
    LightningCLI,
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoModel, AutoTokenizer

import data_modules
import models
from my_dataclasses import IntentCsvData

MODEL_REGISTRY.register_classes(models, LightningModule)
DATAMODULE_REGISTRY.register_classes(data_modules, LightningDataModule)


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        data_params = self.config["fit"]["model"]
        model_params = self.config["fit"]["data"]
        trainer_params = self.config["fit"]["trainer"]
        hparams_dict = {
            "model": model_params["class_path"],
            "epochs": trainer_params["max_epochs"],
        }

        self.trainer.logger.log_hyperparams(hparams_dict)


# parser.add_argument("--plot_name", default="plot_embeddings")

if __name__ == "__main__":

    # cli = MyLightningCLI()
    # model_name = "bert_base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = None
    # # model = CrossEncoder("distilroberta-base", num_labels=1)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # dm = data_modules.IntentDataModule(tokenizer=tokenizer)
    # dm.setup()
    # dl = dm.train_dataloader()
    # data = [dl.dataset[0]]
    # # model.eval()
    # embeds = model.encode(data[0])
    # cli = LightningCLI()
    cli = MyLightningCLI()
    # a = 1
