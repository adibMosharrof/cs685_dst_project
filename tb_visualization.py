import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tbparse import SummaryReader


class TbVisualization:
    def __init__(
        self,
        versions,
        metrics,
        log_dir="lightning_logs/",
        out_dir="visualizations",
        title="",
    ):
        self.versions = versions
        self.title = title
        self.metrics = metrics
        project_dir = Path(os.path.realpath(__file__)).parent
        self.log_dir = project_dir / Path(log_dir)
        self.out_dir = project_dir / Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.acc_graph_name = "accuracy.png"
        plt.style.use("seaborn-darkgrid")
        plt.set_cmap(mpl.cm.viridis)

    def plot_metrics(self):
        data = {m: [] for m in self.metrics}
        labels = []

        for v in self.versions:
            tb = SummaryReader(self.log_dir / f"version_{v}", pivot=True)
            for m in self.metrics:
                if m not in tb.scalars.columns:
                    continue
                vals = tb.scalars[m][tb.scalars[m].notnull()].tolist()
                data[m].append(vals)
            labels.append(m)

        fig, axis = plt.subplots()
        fig.suptitle(self.title)
        axis.set_ylabel("accuracy")
        for l, d in data.items():
            axis.plot(range(0, len(d[0])), d[0], label=l)
        fig.legend()
        path = self.out_dir / f"{self.title}.png"
        fig.savefig(path)


@hydra.main(config_path="config/tb_configs", config_name="intent_vis")
def hydra_start(cfg: DictConfig) -> None:
    tbv = TbVisualization(
        cfg.versions,
        cfg.metrics,
        log_dir=cfg.log_dir,
        title=cfg.plot_title,
    )
    tbv.plot_metrics()


if __name__ == "__main__":
    hydra_start()
