import os
import logging
import shutil as sh
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import numpy as np
import albumentations as A
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, Subset
from endoanalysis.datasets import PointsDataset
from endoanalysis.targets import Keypoints, keypoints_list_to_batch
from endoanalysis.visualization import visualize_keypoints
from neurone.data.heatmaps import make_heatmap
from neurone.data.keypoints import rescale_keypoints
from neurone.utils.general import write_yaml, load_yaml


def collate_im_kp_hm(samples):

    images = [x["image"] for x in samples]
    keypoints_groups = [x["keypoints"] for x in samples]
    heatmaps = [x["heatmaps"] for x in samples]

    return_dict = {
        "image": torch.stack(images, 0).contiguous(),
        "keypoints": keypoints_list_to_batch(keypoints_groups),
        "heatmaps": torch.stack(heatmaps, 0).contiguous(),
    }

    return return_dict


class HeatmapsDataset(PointsDataset, Dataset):
    "Dataset with images, keypoints and the heatmaps corresdonding to them."

    def __init__(
        self,
        images_list,
        labels_list,
        class_labels_map={},
        model_in_channels=3,
        normalization=None,
        resize_to=None,
        sigma=1,
        augs_list=[],
        heatmaps_shape=None,
        class_colors={0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)},
    ):

        super(HeatmapsDataset, self).__init__(
            images_list=images_list, labels_list=labels_list
        )

        self.sigma = sigma

        self.num_classes = len(set(class_labels_map.values()))
        self.class_labels_map = class_labels_map

        self.heatmaps_shape = heatmaps_shape
        self.normalization = normalization

        self.model_in_channels = model_in_channels

        if resize_to:
            self.resize_transform = A.augmentations.Resize(*resize_to)
            augs_list.append(self.resize_transform)
        self.augs_list = augs_list

        self.enable_augs()
        self.class_colors = class_colors
        self.heatmap_bell = self.create_heatmap_bell()

    def enable_augs(self):
        self.alb_transforms = A.Compose(
            self.augs_list,
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )

    def disable_augs(self):
        self.alb_transforms = A.Compose(
            [self.resize_transform],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            ),
        )
    def create_heatmap_bell(self):
        window_size = np.round(4 * self.sigma).astype(int)
        if window_size %2 == 0:
            window_size += 1
        center_i = int((window_size -1 )/ 2)

        ys, xs = np.indices((window_size, window_size))
        ys -= center_i
        xs -= center_i
        base_bell = np.exp(-(ys * ys + xs * xs) / (2 * self.sigma**2))
        return base_bell
        
    def __getitem__(self, x):

        sample = super(HeatmapsDataset, self).__getitem__(x)
        # sample["image_path"] = self.images_paths[x]
        # sample["labels_path"] = self.labels_paths[x]
        
        y_size, x_size, _ = sample["image"].shape

        if self.alb_transforms is not None:

            keypoints_no_class = np.stack(
                [sample["keypoints"].x_coords(), sample["keypoints"].y_coords()]
            ).T
            classes = list(sample["keypoints"].classes())

            transformed = self.alb_transforms(
                image=sample["image"],
                keypoints=keypoints_no_class,
                class_labels=classes,
            )

            kp_coords = np.array(transformed["keypoints"])
            classes = np.array(transformed["class_labels"]).reshape(-1, 1)

            sample["keypoints"] = Keypoints(
                np.hstack([kp_coords, classes]).astype(float)
            )
            sample["image"] = transformed["image"]

        if self.class_labels_map:
            labels_to_keep = self.class_labels_map.keys()
            kp_filtered = []
            for class_label in labels_to_keep:
                kp_class = sample["keypoints"][
                    sample["keypoints"].classes() == float(class_label)
                ]
                new_classes = (
                    np.ones(len(kp_class)) * self.class_labels_map[class_label]
                )
                kp_class = Keypoints(
                    np.vstack([kp_class.x_coords(), kp_class.y_coords(), new_classes]).T
                )
                kp_filtered.append(kp_class)

            sample["keypoints"] = Keypoints(np.vstack(kp_filtered))

        if self.heatmaps_shape:
            keypoints_to_heatmap = rescale_keypoints(
                sample["keypoints"], sample["image"].shape, self.heatmaps_shape
            )
            y_size, x_size = self.heatmaps_shape
        else:
            keypoints_to_heatmap = sample["keypoints"]

        sample["heatmaps"] = make_heatmap(
            x_size, y_size, keypoints_to_heatmap, self.num_classes, self.heatmap_bell
        )


        sample["image"] = np.moveaxis(sample["image"], -1, 0)

        for key in ["heatmaps", "image"]:
            sample[key] = torch.tensor(sample[key]).float()

        if self.normalization:
            sample["image"] -= torch.tensor(self.normalization["mean"]).reshape(
                -1, 1, 1
            )
            sample["image"] /= torch.tensor(self.normalization["std"]).reshape(-1, 1, 1)

        if self.model_in_channels == 1:
            sample["image"] = sample["image"].mean(axis=0)[np.newaxis]

        return sample

    def collate(self, samples):
        return collate_im_kp_hm(samples)

    def visualize(
        self,
        x,
        show_labels=True,
        labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
    ):

        sample = self[x]
        if self.normalization:
            sample["image"] = sample["image"] * torch.tensor(
                self.normalization["std"]
            ).view(-1, 1, 1) + torch.tensor(self.normalization["mean"]).view(-1, 1, 1)
        sample["image"] = sample["image"].int().numpy()
        sample["image"] = np.moveaxis(sample["image"], 0, -1)

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            sample["image"],
            keypoints,
            class_colors=self.class_colors,
            circles_kwargs=labels_kwargs,
        )


class ConcatCollaterDataset(ConcatDataset):
    def collate(self, samples):
        return collate_im_kp_hm(samples)

class SubsetCollater(Subset):
    def collate(self, samples):
        return collate_im_kp_hm(samples)

class Precompution:
    """
    Precomputes dataset with augmentations.

    Precompution take the Dataset and output dir, in which dataset will be saved.
    Precompution iters throught entire Dataset several times and saves
    all images, keypoints and heatmaps as files in outputdir.

    Parameters
    ----------
    dataset: Dataset
        the instance of the Dataset class.
    output_dir: str
        path to the output dir.
    repeats_num: int
        number times each sample is saved called and saved.
    overwrite: bool
        the flag of overwriting files.
    num_workers: int
        number of workers for multiprocessing.
    data_fields: list of str.
        data_fields to save. Must coincide with the keys of samples from the dataset.
    add_sample_with_no_aug: bool
        add samples withou augmantations.
    data_filename: str
        filename of yml file with dataset specs
    jupyter_mode: bool
        whether to run in jupyer. Affects only pbar settings.
    """

    def __init__(
        self,
        dataset,
        output_dir,
        repeats_num=3,
        overwrite=False,
        num_workers=1,
        check_data={},
        data_fields=["image", "keypoints", "heatmaps"],
        add_sample_with_no_aug=False,
        data_filename="precomp_data",
        with_index=True,
        jupyter_mode=False,
    ):

        self.dataset = dataset
        self.output_dir = output_dir
        self.repeats_num = repeats_num
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.data_fields = data_fields
        self.data_filename = data_filename
        self.num_samples = len(self.dataset)

        self.check_data = check_data
        self.check_data["num_samples"] = self.num_samples
        self.check_data["num_repeats"] = self.repeats_num
        self.add_sample_with_no_aug = add_sample_with_no_aug
        self.with_index = with_index

        if jupyter_mode:
            self.tqdm_class = tqdm_notebook
        else:
            self.tqdm_class = tqdm

        self._data_savers = {
            "image": self._save_torch,
            "keypoints": self._save_numpy,
            "heatmaps": self._save_torch,
            "heatmaps_pred": self._save_torch,
            "confidences": self._save_torch,
        }

    def _save_torch(self, x, file_path_no_extention):
        """
        Saves torch tensor with "pt" extention
        """
        torch.save(x, f"{file_path_no_extention}.pt")

    def _save_numpy(self, x, file_path_no_extention):
        """
        Saves numpy array with "npy" extention
        """
        np.save(f"{file_path_no_extention}.npy", x)

    def _write_sample(self, sample, index, repeat_tag):

        sample_fields = sample.keys()

        if set(sample_fields) != set(self.data_fields):
            message = " ".join(
                [
                    "Data fields from the dataset are not coinciding with the required ones.",
                    "\nDataset fields:   ",
                    " ".join(sample_fields),
                    "\nRequired data fields:   ",
                    " ".join(self.data_fields),
                ]
            )
            raise Exception(message)

        for data_field in self.data_fields:
            file_path = os.path.join(
                self.output_dir, repeat_tag, data_field, str(index)
            )
            self._data_savers[data_field](sample[data_field], file_path)

    def _process_sample(self, index):
        """
        Process a single sample.

        Parameters
        ----------
        index: int
            index of sample from self.dataset
        """

        if self.add_sample_with_no_aug:
            self.dataset.disable_augs()
            sample = self.dataset[index]
            self.dataset.enable_augs()
            self._write_sample(sample, index, "no_augs")

        for i in range(self.repeats_num):
            sample = self.dataset[index]
            self._write_sample(sample, index, str(i))
            
    
    def _create_index(self):
        if self.with_index:
            index = {
                "images": {x: self.dataset.images_paths[x] for x in range(len(self.dataset))},
                "labels": {x: self.dataset.images_paths[x] for x in range(len(self.dataset))}
            }
            write_yaml(
                os.path.join(self.output_dir, "index.yml"),
                index
            )

    def make(self):
        """
        Iter through dataset and save all features to the output dir.
        """
        self._make_dir(self.output_dir)
        self._create_index()

        repeat_tags = [str(x) for x in range(self.repeats_num)]
        if self.add_sample_with_no_aug:
            repeat_tags.append("no_augs")

        for tag in repeat_tags:
            self._make_dir(os.path.join(self.output_dir, tag))
            for data_field in self.data_fields:
                self._make_dir(os.path.join(self.output_dir, tag, data_field))

        logging.info("Precomputing... ")
        if self.num_workers == 1:
            for index in self.tqdm_class(range(len(self.dataset))):
                self._process_sample(index)
        else:
            with Pool(self.num_workers) as pool:
                for _ in self.tqdm_class(
                    pool.imap(self._process_sample, range(self.num_samples)),
                    total=self.num_samples,
                ):
                    pass

        write_yaml(
            f"{os.path.join(self.output_dir, self.data_filename)}.yaml",
            {
                "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                "data_fields": self.data_fields,
                "check_data": self.check_data,
                "samples_with_no_aug": self.add_sample_with_no_aug,
            },
        )

        tqdm.write("Done!")

    def _make_dir(self, dir):
        """
        Making new directory.

        Parameters
        ----------
        dir: str
            path to directory.
        """
        if os.path.isdir(dir):
            if self.overwrite:
                sh.rmtree(dir)
            else:
                raise Exception(
                    "Output directory is not empty and overwrite flag is disabled, aborting."
                )
        os.makedirs(dir)


class PrecomputedDataset(HeatmapsDataset):
    """
    Dataset which loads the precomputed images, keypoints and heatmaps.

    Parameters
    ----------
    dataset_dir: str
        path to the dir with precomputed dataset
    repeats_num: int
        number of times the precomputed dataset is sampled
    check_data: dict
        metadata for the checking precomputed dataset
    """

    def __init__(self, data_file_path, check_data=None):

        config = load_yaml(data_file_path)
        if check_data is not None and config["check_data"] != check_data:

            logging.error("Precomputed:")
            logging.error(config["check_data"])
            logging.error("Expected:")
            logging.error(check_data)

            raise ValueError("Data check not passed in precomputed dataset.")
        self.num_samples = config["check_data"]["num_samples"]
        self.root_dir = os.path.split(data_file_path)[0]
        self.use_augs = True
        self.samples_with_no_aug = config["samples_with_no_aug"]
        self.data_fields = config["data_fields"]
        self.repeats_ids = [i for i in range(config["check_data"]["num_repeats"])]

        self._data_openers = {
            "image": self._load_torch,
            "keypoints": self._load_numpy,
            "heatmaps": self._load_torch,
            "heatmaps_pred": self._load_torch,
            "confidences": self._load_torch,
        }

        self.collaters = {
            "image": self._torch_collater,
            "keypoints": keypoints_list_to_batch,
            "heatmaps": self._torch_collater,
            "heatmaps_pred": self._torch_collater,
            "confidences": self._torch_collater,
        }

        self.enable_augs()

    def disable_augs(self):
        if not self.samples_with_no_aug:
            raise Exception(
                "Cannnot disable augmentations in precomputed dataset: samples_with_no_aug is False"
            )
        else:
            self.use_augs = False

    def enable_augs(self):
        self.use_augs = True

    def _load_torch(self, file_path_no_extention):
        """
        Loads torch tensor with a pt extention
        """
        return torch.load(f"{file_path_no_extention}.pt")

    def _load_numpy(self, file_path_no_extention):
        """
        Loads numpy array with a npy extention
        """
        return Keypoints(np.load(f"{file_path_no_extention}.npy"))

    def _torch_collater(self, list_of_tensors):
        """Collates torch tensors"""
        return torch.stack(list_of_tensors, 0).contiguous()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_augs:
            repeat_tag = str(random.choice(self.repeats_ids))
        else:
            repeat_tag = "no_augs"

        sample = {}
        for data_field in self.data_fields:
            path = os.path.join(self.root_dir, repeat_tag, data_field, str(idx))
            sample[data_field] = self._data_openers[data_field](path)

        return sample

    def collate(self, samples):

        batch = {}
        for field in self.data_fields:
            batch[field] = self.collaters[field]([x[field] for x in samples])

        return batch


class HeatmapModelDataset(Dataset):
    """Dataset to handle pregenerated heatmaps"""

    def __init__(self, dataset, model, device=torch.device("cpu"), images_ids=None):

        self.images_ids = images_ids
        self.dataset = dataset
        self.model = model
        self.device = device

    def __len__(self):
        if self.images_ids is None:
            return len(self.dataset)
        else:
            return len(self.images_ids)

    def __getitem__(self, x):

        if self.images_ids is not None:
            idx = self.images_ids[x]
        else:
            idx = x

        sample = self.dataset[idx]
        batch = self.dataset.collate([sample])
        image_batched = batch["image"].to(self.device)
        self.model.eval()
        with torch.no_grad():
            heatmaps, _, confidences = self.model(image_batched)
        sample.pop("heatmaps", None)
        sample.pop("image", None)
        sample["heatmaps_pred"] = heatmaps.detach().cpu()[0]

        return sample

    def collate(self, samples):

        keypoints_groups = [x["keypoints"] for x in samples]
        heatmaps = [x["heatmaps_pred"] for x in samples]

        return_dict = {
            "keypoints": keypoints_list_to_batch(keypoints_groups),
            "heatmaps_pred": torch.stack(heatmaps, 0).contiguous().to(self.device),
        }

        return return_dict
