import os
import yaml
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from endoanalysis.datasets import parse_master_yaml, PointsDataset
from neurone.data.datasets import PrecomputedDataset
from neurone.utils.general import makedir_overwrite, write_yaml


def compose_fold(target_dir, fold_name, images_paths, labels_paths):
    """
    Creates one fold for a split

    Parameters
    ----------
    target_dir: str
        path to directory where all the folds are stored
    fold_name: str
        name of the fold
    images_paths: list of str
        paths to the fold's images
    labels_paths: list of str
        paths to the fold's labels files. Must be in correspondence with images_paths
    """

    fold_dir_path = os.path.join(target_dir, fold_name)
    lists_dir_path = os.path.join(fold_dir_path, "lists")
    os.makedirs(fold_dir_path, exist_ok=True)
    os.makedirs(lists_dir_path, exist_ok=True)
    images_list_path = os.path.join(lists_dir_path, "images.txt")
    labels_list_path = os.path.join(lists_dir_path, "labels.txt")
    yaml_path = os.path.join(fold_dir_path, ".".join([fold_name, "yaml"]))

    fold_images_paths = []
    fold_labels_paths = []
    tqdm.write("Creating %s... " % os.path.basename(fold_dir_path), end="")
    for image_path, labels_path in zip(images_paths, labels_paths):
        image_path = os.path.normpath(os.path.relpath(image_path, start=lists_dir_path))
        labels_path = os.path.normpath(
            os.path.relpath(labels_path, start=lists_dir_path)
        )
        fold_images_paths.append(image_path + "\n")
        fold_labels_paths.append(labels_path + "\n")

    with open(images_list_path, "w+") as file:
        file.writelines(fold_images_paths)

    with open(labels_list_path, "w+") as file:
        file.writelines(fold_labels_paths)

    with open(yaml_path, "w+") as file:
        yaml.safe_dump(
            {
                "images_lists": [
                    os.path.relpath(images_list_path, start=fold_dir_path)
                ],
                "labels_lists": [
                    os.path.relpath(labels_list_path, start=fold_dir_path)
                ],
            },
            file,
        )
    tqdm.write("Done!")


def get_pseudoclasses(dataset, num_classes, jupyter_mode=False):
    """
    Assignes image-level pseudoclasses based on keypoiunts number.

    Parameters
    ----------
    dataset: endoanalysis.datasets.PointsDataset
        dataset to take images and keypoints from
    num_classes: int
        total number of classes
    jupyter_mode: bool
        whether to use in jupyter or not
    Returns
    -------
    pseudoclasses: ndarray of int
        pseudoclasses for images from the dataset
    """

    if jupyter_mode:
        tqdm_to_use = tqdm_notebook
    else:
        tqdm_to_use = tqdm
    tqdm_to_use.write("Creating pseudoclases")
    clases_stats = np.zeros((len(dataset), num_classes))
    for i, sample in tqdm_to_use(enumerate(dataset), total=len(dataset)):
        classes = sample["keypoints"].classes()
        clases_stats[i] = np.sum(
            classes.reshape(-1, 1) == np.arange(num_classes), axis=0
        )
    tqdm_to_use.write("Done!")
    pseudoclasses = clases_stats.argmax(axis=1)

    return pseudoclasses


def make_kfold(
    master_yaml, target_dir, num_folds, num_classes, pseudostrat=True, overwrite=False
):
    """
    Makes train-test split with creating new files lists and master yamls.
    The split is perofmed in quasi-stratified manner.

    Parameters
    ----------
    master_yaml: str
        path to master_yaml. Only the files from the lists from it will be considered in splits.
    target_dir: str
        path to target dir to store the split
    num_classes: int
        number of keypoints classes
    train_size: float
        fraction of samples which go to train. Should be between 0. and 1.
    pseudostrat: bool
        whether to make stratification with pseudoclasses
    overwrite: bool
         whether the to delete target_dir if it exists
    """

    makedir_overwrite(target_dir, overwrite=overwrite)

    train_lists = parse_master_yaml(master_yaml)
    dataset = PointsDataset(train_lists["images_lists"], train_lists["labels_lists"])

    images_ids = np.arange(len(dataset))

    if pseudostrat:
        pseudoclasses = get_pseudoclasses(dataset, num_classes)
        skfold = StratifiedKFold(n_splits=num_folds)
        split = skfold.split(images_ids, pseudoclasses)
    else:
        kfold = KFold(n_splits=num_folds)
        split = kfold.split(images_ids)

    for fold_i, (_, fold_ids) in enumerate(split):
        images_paths = [dataset.images_paths[x] for x in fold_ids]
        labels_paths = [dataset.labels_paths[x] for x in fold_ids]
        fold_name = "_".join(["fold", str(fold_i)])
        compose_fold(target_dir, fold_name, images_paths, labels_paths)

    write_yaml(
        os.path.join(target_dir, "split_info.yml"),
        {
            "split_type": "kfold",
            "is_precomp": False,
            "num_folds": num_folds,
            "pseudostrat": pseudostrat,
            "num_classes": num_classes,
        },
    )

def split_trainval_ids(dataset, num_classes, train_size, pseudostrat):
    """Get train and val ids for trainval_split.

    Parameters
    ----------
    dataset: endoanalysis.datasets.PointsDataset
        dataset to split
    num_classes: int
        number of keypoints classes
    train_size: float
        fraction of samples which go to train. Should be between 0. and 1.
    pseudostrat: bool
        whether to make stratification with pseudoclasses
    Returns
    -------
        train_ids: list of int
            ids for training
        val_isd: list of int
            ids for validation
    """
    if pseudostrat:
        pseudoclasses = get_pseudoclasses(dataset, num_classes)
        train_ids, val_ids = train_test_split(
            np.arange(len(dataset)), train_size=train_size, stratify=pseudoclasses
        )
    else:
        train_ids, val_ids = train_test_split(
            np.arange(len(dataset)), train_size=train_size
        )
        
    return train_ids, val_ids
    

def make_train_val_split(
    master_yaml, target_dir, train_size, num_classes, pseudostrat=True, overwrite=False
):
    """
    Makes train-test split with creating new files lists and master yamls.
    The split is performed in quasi-stratified manner.

    Parameters
    ----------
    master_yaml: str
        path to master_yaml. Only the files from the lists from it will be considered in splits.
    target_dir: str
        path to target dir to store the split
    num_classes: int
        number of keypoints classes
    train_size: float
        fraction of samples which go to train. Should be between 0. and 1.
    pseudostrat: bool
        whether to make stratification with pseudoclasses
    overwrite:
         whether the to delete target_dir if it exists
    """

    makedir_overwrite(target_dir, overwrite=overwrite)

    train_lists = parse_master_yaml(master_yaml)
    dataset = PointsDataset(train_lists["images_lists"], train_lists["labels_lists"])

    train_ids, val_ids = split_trainval_ids(dataset, num_classes, train_size, pseudostrat)

    for subdir_name, ids_list in [("train", train_ids), ("val", val_ids)]:
        images_paths = [dataset.images_paths[x] for x in ids_list]
        labels_paths = [dataset.labels_paths[x] for x in ids_list]
        compose_fold(target_dir, subdir_name, images_paths, labels_paths)

    write_yaml(
        os.path.join(target_dir, "split_info.yml"),
        {
            "split_type": "trainval",
            "is_precomp": False,
            "train_size": train_size,
            "pseudostrat": pseudostrat,
            "num_classes": num_classes,
        },
    )
