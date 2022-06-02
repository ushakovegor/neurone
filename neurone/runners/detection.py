import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pprint import pprint
from torch.utils.data import DataLoader

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import EngineCheckpointerCallback
from animus.torch.engine import CPUEngine, DDPEngine, DPEngine, GPUEngine, XLAEngine
from neurone.utils.configs import model_from_config, parse_train_config, parse_eval_config
from neurone.core.train.procedures import compose_split_datasets


class TestExperiment(IExperiment):
    def __init__(self,
                 engine,
                 config={}):

        super().__init__()
        self.engine = engine 
        self.config = config

        self.batch_size = config["train"]["batch_size"]
        self.num_epochs = self.config["train"]["num_epochs"]
        self.workers = config["data"]["workers"]
        self.log_every = self.config["train"]["log_every"]
        self.checkpoint_every = self.config["train"]["checkpoint_every"]
        self.overwrite = self.config["data"]["overwrite"]
        self.save_dir = self.config["data"]["model_dir"]
        # self. =

    def _setup_data(self):
        datasets = compose_split_datasets(self.config)
        datasets["train"].enable_augs()
        datasets["valid"].disable_augs()
        train_loader = DataLoader(
            datasets["train"],
            collate_fn=datasets["train"].collate,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )
        valid_loader = DataLoader(
            datasets["valid"],
            collate_fn=datasets["valid"].collate,
            batch_size=self.batch_size,
            num_workers=self.workers
        )
    
        if datasets.get("test") is None:
            test_loader = valid_loader
        else:
            test_loader = DataLoader(
                datasets.get("test"),
                collate_fn=datasets.get("test").collate,
                batch_size=self.batch_size,
                num_workers=self.workers
            )
        train_loader, valid_loader, test_loader = self.engine.prepare(train_loader, valid_loader, test_loader)
        self.datasets = {"train": train_loader, "valid": valid_loader, "test": test_loader}


    def _setup_model(self):
        self.model = model_from_config(self.config["model"]).to(self.engine.device)
        self.criterion, self.optimizer, self.scheduler, self.train_metric, self.val_metric = parse_train_config(
            self.config["train"], self.model)
        self.eval_metrics = parse_eval_config(self.config["eval"])
        self.model, self.optimizer = self.engine.prepare(
            self.model, self.optimizer)
        self.model, self.optimizer = self.engine.prepare(self.model, self.optimizer)
        

    # def _setup_callbacks(self):
    #     self.callbacks = {
    #         "early-stop": EarlyStoppingCallback(
    #             minimize=False,
    #             patience=5,
    #             dataset_key="valid",
    #             metric_key="accuracy",
    #             min_delta=0.01,
    #         ),
    #         "checkpointer": EngineCheckpointerCallback(
    #             exp_attr="model",
    #             logdir="./logs_torch_dl",
    #             dataset_key="valid",
    #             metric_key="accuracy",
    #             minimize=False,
    #         ),
    #     }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self.engine.setup(self._local_rank, self._world_size)
        with self.engine.local_main_process_first():
            self._setup_data()
        self._setup_model()
        # self._setup_callbacks()

    def run_dataset(self):
        total_loss, total_accuracy = 0.0, 0.0

        self.model.train(self.is_train_dataset)
        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, batch in enumerate(
                tqdm(self.dataset, disable=not self.engine.is_local_main_process)
            ):
                self.optimizer.zero_grad()
                output = self.model(batch["image"])
                loss = self.criterion(output, batch["heatmaps"])
                total_loss += loss.sum().item()
                # add accuracy
                if self.is_train_dataset:
                    self.engine.backward(loss)
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step
        total_accuracy /= self.dataset_batch_step * self.batch_size
        self.dataset_metrics = {"loss": total_loss, "accuracy": total_accuracy}
        self.dataset_metrics = self.engine.mean_reduce_ddp_metrics(
            self.dataset_metrics)
        self.dataset_metrics = {k: float(v)
                                for k, v in self.dataset_metrics.items()}

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        if self.engine.is_local_main_process:
            pprint(self.epoch_metrics)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self.engine.cleanup()

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._local_rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")

    def _run(self) -> None:
        self.engine.spawn(self._run_local)
