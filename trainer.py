from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import (
    DATAMODULE_REGISTRY,
    MODEL_REGISTRY,
    LightningArgumentParser,
    LightningCLI,
)

import data_modules
import models

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


if __name__ == "__main__":

    cli = MyLightningCLI()
