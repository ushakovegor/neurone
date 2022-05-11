from logging import warning
import os
import datetime
import shutil as sh
from tqdm import tqdm, tqdm_notebook
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from nucleidet.train.utils import weight_init, make_checkpoint
from nucleidet.utils.general import makedir_overwrite, write_yaml


class HeatmapTrainer:
    def __init__(
        self,
        detector,
        epoch_num,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        scheduler=None,
        config={},
        train_metrics=[],
        val_metrics=[],
        save_best=[],
        device=torch.device("cpu"),
        log_every=1,
        best_checkpoints_from=1,
        checkpoint_every=1,
        overwrite=False,
        save_dir="",
        init_weights="",
        jupyter_mode=False,
    ):

        self.detector = detector
        self.epoch_num = epoch_num
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        for metric in train_metrics + val_metrics:
            if metric.name == "train_loss":
                raise ValueError("No metric can have the name 'train_loss")
            if metric.name_group == "train_loss":
                raise ValueError("No metric can have the name group 'train_loss")

        self.save_best = save_best
        self.device = device

        self.save_dir = save_dir
        self.init_weights = init_weights
        self.overwrite = overwrite
        self.log_every = log_every
        self.best_checkpoints_from = best_checkpoints_from
        self.checkpoint_every = checkpoint_every

        if log_every >= epoch_num:
            message = "".join(
                [
                    "The logging will be done every %ith epoch " % log_every,
                    "while the total number of epochs is %i. " % epoch_num,
                    "Best models will not be saved.",
                ]
            )
            warning(message)

        if jupyter_mode:
            self.tqdm_pbar = tqdm_notebook
        else:
            self.tqdm_pbar = tqdm

    def run_batch(self, batch):
        images = batch["image"].to(self.device)
        heatmaps_pred, keypoints, confidences = self.detector(images)
        batch_pred = {
            "heatmaps": heatmaps_pred,
            "keypoints": keypoints,
            "confidences": confidences,
        }

        return batch_pred

    def evaluate(self, epoch_i, writer, log_dir):
        self.detector.eval()
        for metric in self.val_metrics:
            metric.reset()

        for batch in self.val_loader:
            with torch.no_grad():
                batch_pred = self.run_batch(batch)
            for metric in self.val_metrics:
                metric.update(batch, batch_pred)

        for metric in self.val_metrics:
            metric.compute()
            self.log_metric(metric, epoch_i, log_dir, writer)

    def prepare_for_train(self):

        makedir_overwrite(self.save_dir, self.overwrite)
        self.detector.set_predict_keypoints(False)
        dirs = {
            "tensorboard": os.path.join(self.save_dir, "tensorboard"),
            "checkpoints": os.path.join(self.save_dir, "checkpoints"),
            "checkpoints_regular": os.path.join(
                self.save_dir, "checkpoints", "regular"
            ),
            "checkpoints_featured": os.path.join(
                self.save_dir, "checkpoints", "featured"
            ),
            "logs": os.path.join(self.save_dir, "logs"),
            "train_logs": os.path.join(self.save_dir, "logs", "train"),
            "val_logs": os.path.join(self.save_dir, "logs", "val"),
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path)

        write_yaml(os.path.join(self.save_dir, "config.yml"), self.config)

        with open(os.path.join(self.save_dir, "time.txt"), "w+") as file:
            file.write(datetime.datetime.now().strftime("%H:%M %d.%m.%y"))

        if self.init_weights:
            self.detector.load_state_dict(self.init_weights)
        else:
            self.detector.apply(weight_init)

        self.detector.train()
        self.detector.to(self.device)
        self.criterion.to(self.device)

        writer = SummaryWriter(dirs["tensorboard"])

        return dirs, writer

    def checkpoint_best_metrics(self, epoch_i, best_metrics_values, checkpoints_dir):

        for metric in self.train_metrics + self.val_metrics:
            if metric.name in self.save_best:
                new_metric_value = metric.min_max_factor() * metric.get_value()
                if (
                    new_metric_value
                    >= best_metrics_values[metric.name] * metric.min_max_factor()
                ):
                    best_metrics_values[metric.name] = (
                        new_metric_value * metric.min_max_factor()
                    )
                    make_checkpoint(
                        self.detector,
                        self.optimizer,
                        self.scheduler,
                        epoch_i,
                        os.path.join(
                            checkpoints_dir, "".join(["best_", metric.name, ".pth"])
                        ),
                        overwrite=True,
                    )

        return best_metrics_values

    def log_metric(self, metric, epoch_i, log_dir, writer):

        name, name_group, value, precision = metric.get_logging_info()

        writer.add_scalars(name_group, {name: value}, epoch_i)

        file_path = os.path.join(log_dir, ".".join([name, "csv"]))
        if os.path.exists(file_path):
            values = np.loadtxt(file_path)
            values = np.append(values, np.round(value, decimals=precision))
        else:
            values = np.array([value])

        fmt = "%" + "." + str(precision) + "f"
        np.savetxt(file_path, values, delimiter=",", fmt=fmt)

    def train(self):

        tqdm.write("Heatmap training initiated:")
        dirs, writer = self.prepare_for_train()
        best_metrics_values = {x: np.inf for x in self.save_best}

        for epoch_i in self.tqdm_pbar(
            range(self.epoch_num), desc="Experiment", position=0, leave=True
        ):
            train_losses = []

            if epoch_i % self.log_every == 0:
                for metric in self.train_metrics:
                    metric.reset()

            self.detector.train()
            for batch in self.tqdm_pbar(
                self.train_loader, desc="Epoch #%i progress" % epoch_i, leave=False
            ):

                batch_pred = self.run_batch(batch)
                heatmaps_gt = batch["heatmaps"].to(self.device)
                loss = self.criterion(heatmaps_gt, batch_pred["heatmaps"])
                loss.backward()
                train_losses.append(loss.data.item())
                self.optimizer.step()
                self.optimizer.zero_grad()

                if epoch_i % self.log_every == 0:
                    for metric in self.train_metrics:
                        metric.update(batch, batch_pred)

            if not self.scheduler is None:
                self.scheduler.step()

            train_loss = np.average(train_losses)
            tqdm.write("Epoch #%i train loss: %0.8f" % (epoch_i, train_loss))
            writer.add_scalar("train_loss", train_loss, epoch_i)

            if epoch_i % self.log_every == 0:
                for metric in self.train_metrics:
                    metric.compute()
                    self.log_metric(metric, epoch_i, dirs["train_logs"], writer)

                self.evaluate(epoch_i, writer, dirs["val_logs"])

                if epoch_i >= self.best_checkpoints_from:
                    best_metrics_values = self.checkpoint_best_metrics(
                        epoch_i,
                        best_metrics_values,
                        dirs["checkpoints_featured"],
                    )

            if epoch_i % self.checkpoint_every == 0 and epoch_i != 0:
                make_checkpoint(
                    self.detector,
                    self.optimizer,
                    self.scheduler,
                    epoch_i,
                    os.path.join(
                        dirs["checkpoints_regular"],
                        "".join(["epoch_", str(epoch_i), ".pth"]),
                    ),
                    overwrite=False,
                )

        make_checkpoint(
            self.detector,
            self.optimizer,
            self.scheduler,
            epoch_i,
            os.path.join(dirs["checkpoints_featured"], "last.pth"),
            overwrite=False,
        )
        self.detector.set_predict_keypoints(True)
        tqdm.write("Heatmap training finished.")
