import logging
import os
import shutil as sh
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
from endoanalysis.similarity import KPSimilarity
from nucleidet.data.datasets import HeatmapModelDataset
from nucleidet.data.datasets import Precompution, PrecomputedDataset
from nucleidet.train.meters import mAPmetric
from nucleidet.utils.general import write_yaml, load_yaml


class KeypointsExtractorTrainer:
    def __init__(
        self,
        detector,
        datastet,
        model_dir,
        checkpoint_type,
        num_iters,
        similarity_scale,
        sim_thresh,
        params_bounds,
        num_init_keypoints,
        batch_size=1,
        num_workers=1,
        device=torch.device("cpu"),
        jupyter_mode=False,
        classes_to_consider=range(2),
    ):

        self.detector = detector
        self.dataset = datastet
        self.model_dir = model_dir
        self.checkpoint_type = checkpoint_type
        self.device = device
        self.num_workers = num_workers
        self.jupyter_mode = jupyter_mode

        self.extractor = detector.keypoints_extractor

        self.optimizer = BayesianOptimization(
            f=self.calculate_map,
            pbounds=params_bounds,
        )

        self.meter = mAPmetric(
            KPSimilarity(scale=similarity_scale), classes_to_consider, sim_thresh
        )

        self.num_init_points = num_init_keypoints
        self.num_iters = num_iters
        self.batch_size = batch_size

    def params_for_extractor(
        self, half_pooling_scale, min_peak_value, supression_range
    ):
        params = {
            "pooling_scale": int(2 * np.round(half_pooling_scale) + 1),
            "min_peak_value": float(np.round(min_peak_value, decimals=3)),
            "supression_range": float(np.round(supression_range, decimals=3)),
        }

        return params

    def calculate_map(self, half_pooling_scale, min_peak_value, supression_range):

        self.meter.reset()
        params = self.params_for_extractor(
            half_pooling_scale, min_peak_value, supression_range
        )
        self.extractor.set_params(**params)

        for batch in self.dataloader_for_opt:

            keypoints_pred, confidences = self.extractor(batch["heatmaps_pred"])
            pred_batch = {"keypoints": keypoints_pred, "confidences": confidences}
            self.meter.update(batch, pred_batch)

        self.meter.compute()

        return self.meter.get_value()

    def train(self):
        tqdm.write("Extractor training initiated:")
        checkpoint_path = os.path.join(
            self.model_dir,
            "checkpoints/featured/",
            ".".join([self.checkpoint_type, "pth"]),
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.detector.load_state_dict(checkpoint["model_state_dict"])
        self.detector.to(self.device)
        self.detector.eval()
        model_dataset = HeatmapModelDataset(
            self.dataset, self.detector, device=self.device
        )
        precomputed_heatmaps_dir = os.path.join(self.model_dir, "precomputed_heatmaps")
        precomp = Precompution(
            model_dataset,
            precomputed_heatmaps_dir,
            repeats_num=1,
            num_workers=self.num_workers,
            data_fields=["heatmaps_pred", "keypoints"],
            overwrite=True,
            with_index=False,
            jupyter_mode=self.jupyter_mode,
        )
        logging.info("Saving heatmaps genereated by the model...")
        precomp.make()
        logging.info("Done!")

        dataset_from_model = PrecomputedDataset(
            os.path.join(precomputed_heatmaps_dir, "precomp_data.yaml"),
            check_data={"num_samples": len(model_dataset), "num_repeats": 1},
        )

        self.dataloader_for_opt = DataLoader(
            dataset_from_model,
            collate_fn=dataset_from_model.collate,
            batch_size=self.batch_size,
        )

        logging.info("Optimizing keypoints extractor parameters...")
        self.optimizer.maximize(init_points=self.num_init_points, n_iter=self.num_iters)
        logging.info("Done!")

        sh.rmtree(precomputed_heatmaps_dir)

        self.optimizer.max["params"]
        extractor_params = self.params_for_extractor(**self.optimizer.max["params"])
        self.detector.keypoints_extractor.set_params(**extractor_params)

        write_yaml(
            os.path.join(self.model_dir, "extractor_params.yml"), extractor_params
        )

        config_path = os.path.join(self.model_dir, "config.yml")
        config_backup_path = os.path.join(self.model_dir, "config_backup.yml")
        os.rename(config_path, config_backup_path)
        config = load_yaml(config_backup_path)

        config["model"]["extractor"] = extractor_params
        write_yaml(config_path, config, overwrite=True)
        os.remove(config_backup_path)
        tqdm.write("Extractor training finished.")
