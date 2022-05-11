import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold
import albumentations as A
from endoanalysis.targets import Keypoints
from endoanalysis.datasets import parse_master_yaml, PointsDataset
from nucleidet.models.staining.hscore import calculate_hscore
from nucleidet.utils.general import makedir_overwrite


def check_image_shape(shape):
    if len(shape) != 3 or shape[0] != shape[1]:
        message = "Something is wrong with image shape: "
        message += "(" + ",".join([str(x) for x in shape]) + ")"
        raise Exception(message)


class StainTrainer:
    def __init__(
        self,
        stains_master_yml_path,
        kpStainAnalyzer,
        n_folds,
        image_size,
        save_dir="",
        overwrite=False,
    ):
        lists = parse_master_yaml(stains_master_yml_path)
        self.dataset = PointsDataset(lists["images_lists"], lists["labels_lists"])
        self._tissue_structure_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
        }
        self._stainig_dict = {x: x % 4 for x in range(12)}
        self.kpStainAnalyzer = kpStainAnalyzer
        self.kp_stains, self.kp_structures = self._extract_kp_stains()
        self.n_folds = n_folds
        self.image_size = image_size
        self.save_dir = save_dir
        if save_dir:
            makedir_overwrite(save_dir, overwrite)
            self.scores_dir = os.path.join(save_dir, "scores")
            os.mkdir(self.scores_dir)

    def resize_samples(self):
        resized = []
        transform = A.Compose(
            [
                A.Resize(*self.image_size),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            keypoints_no_class = np.stack(
                [sample["keypoints"].x_coords(), sample["keypoints"].y_coords()]
            ).T
            classes = list(sample["keypoints"].classes())

            transformed = transform(
                image=sample["image"],
                keypoints=keypoints_no_class,
                class_labels=classes,
            )

            kp_coords = np.array(transformed["keypoints"])
            classes = np.array(transformed["class_labels"]).reshape(-1, 1)
            transformed["keypoints"] = Keypoints(
                np.hstack([kp_coords, classes]).astype(float)
            )

            resized.append(transformed)
        return self.dataset.collate(resized)

    def _extract_kp_stains(self):
        kp_classes = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            shape = sample["image"].shape
            check_image_shape(shape)
            kp_classes.append(sample["keypoints"].classes().astype(int))
        kp_classes = np.hstack(kp_classes)
        kp_structures = np.vectorize(self._tissue_structure_dict.get)(kp_classes)
        kp_stains = np.vectorize(self._stainig_dict.get)(kp_classes)
        return kp_stains, kp_structures

    def evaluate_training(self):
        samples = self.resize_samples()
        skf = KFold(n_splits=self.n_folds)
        self.initialize_scores()
        tqdm.write("Evaluation stainer traininng.")
        tqdm.write("Performing kfold evaluation:")
        for train_kp_ids, val_kp_ids in tqdm(
            skf.split(X=samples["keypoints"], y=self.kp_stains), total=self.n_folds
        ):
            self.kpStainAnalyzer.train(
                samples["image"], samples["keypoints"], self.kp_stains, train_kp_ids
            )
            val_keypoints = samples["keypoints"][val_kp_ids]
            val_stains = self.kp_stains[val_kp_ids]
            preds = self.kpStainAnalyzer.get_stains(samples["image"], val_keypoints)
            self.update_scores(val_stains, preds)
        self.output_scores()

        if self.save_dir:
            for score_name, scores in self.scores.items():
                np.savetxt(
                    os.path.join(self.scores_dir, ".".join([score_name, "txt"])), scores
                )

    def initialize_scores(self):
        self.scores = {
            "accuracy": [],
            "f1_scores": [],
            "hscores_gt": [],
            "hscores_pred": [],
        }

    def update_scores(self, val_stains, preds):
        self.scores["accuracy"].append(accuracy_score(val_stains, preds))
        self.scores["f1_scores"].append(f1_score(val_stains, preds, average="micro"))
        self.scores["hscores_gt"].append(calculate_hscore(val_stains))
        self.scores["hscores_pred"].append(calculate_hscore(preds))

    def output_scores(self):
        for key, value in self.scores.items():
            tqdm.write(key + ": " + " ".join(str(np.round(x, 3)) for x in value))
        tqdm.write(
            "MAPE: "
            + str(
                np.round(
                    mean_absolute_percentage_error(
                        self.scores["hscores_gt"], self.scores["hscores_pred"]
                    ),
                    4,
                )
            )
        )
        tqdm.write(
            "MAE: "
            + str(
                np.round(
                    mean_absolute_error(
                        self.scores["hscores_gt"], self.scores["hscores_pred"]
                    ),
                    4,
                )
            )
        )
        tqdm.write(
            "sqrt_MSE: "
            + str(
                np.round(
                    np.sqrt(
                        mean_squared_error(
                            self.scores["hscores_gt"], self.scores["hscores_pred"]
                        )
                    ),
                    4,
                )
            )
        )

    def train(self):
        tqdm.write("Training stainer with the full data... ", end="")
        samples = self.resize_samples()
        self.kpStainAnalyzer.train(
            samples["image"], samples["keypoints"], self.kp_stains
        )
        if self.save_dir:
            self.kpStainAnalyzer.save(os.path.join(self.save_dir, "stainer.z"))
        tqdm.write("Done. ")
