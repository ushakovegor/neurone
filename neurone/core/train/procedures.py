from ast import Sub
import logging
from deepdiff import DeepDiff
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from endoanalysis.datasets import parse_master_yaml
from neurone.utils.configs import model_from_config, albumentations_from_config
from neurone.data.datasets import Precompution, PrecomputedDataset
from neurone.utils.configs import parse_train_config, parse_eval_config
from neurone.data.datasets import HeatmapsDataset, ConcatCollaterDataset, SubsetCollater
from neurone.core.train.heatmap import HeatmapTrainer
from neurone.core.train.extractor import KeypointsExtractorTrainer
from neurone.core.evaluation.evaluator import Evaluator
from neurone.utils.general import makedir_overwrite, load_yaml, write_yaml
from neurone.data.splits import split_trainval_ids


def compose_split_check(model_config, split_info, precomp=False):
    return_dict = {
        "split_type": split_info["split_type"],
        "is_precomp": precomp,
        "storing_method": split_info["storing_method"],
        "pseudostrat": split_info["pseudostrat"],
        "num_precomputions": model_config["train"]["heatmaps"]["num_precomputions"],
        "num_repeats": model_config["train"]["heatmaps"]["num_precomputions"],
        "norm_mean": model_config["model"]["norm_mean"],
        "norm_std": model_config["model"]["norm_std"],
        "image_size": model_config["model"]["image_size"],
        "heatmaps_shape": model_config["model"]["heatmaps_shape"],
        "heatmaps_sigma": model_config["train"]["heatmaps"]["heatmaps_sigma"],
        "classes_to_output": list(model_config["model"]["class_labels_map"].values()),
        "model_in_channels": model_config["model"]["heatmap_model_kwargs"][
            "in_channels"
        ],
    }

    if split_info["split_type"] == "kfold":
        return_dict["num_folds"] = split_info["num_folds"]
    else:
        return_dict["train_size"] = split_info["train_size"]

    return return_dict


def get_heatmaps_dataset(master_yaml_path, config):

    master_yaml = parse_master_yaml(master_yaml_path)
    augs_list = albumentations_from_config(config["train"]["augmentations"])

    return HeatmapsDataset(
        master_yaml["images_lists"],
        master_yaml["labels_lists"],
        class_labels_map=config["model"]["class_labels_map"],
        sigma=config["data"]["heatmaps_sigma"],
        augs_list=augs_list,
        heatmaps_shape=config["data"]["image_shape"],
        normalization={
            "mean": config["data"]["norm_mean"],
            "std": config["data"]["norm_std"],
        },
        model_in_channels=config["model"]["model_kwargs"]["in_channels"],
        resize_to=config["model"]["input_shape"],
    )


def single_trainval_run(
    config,
    train_dataset,
    val_dataset,
    device,
    workers,
    model_dir,
    overwrite,
    test_dataset=None,
    jupyter_mode=False,
):


    
    detector = model_from_config(config["model"])

    (
        criterion,
        optimizer,
        scheduler,
        train_metrics,
        val_metrics,
    ) = parse_train_config(
        config["train"]["heatmaps"],
        detector.heatmap_model,
        device,
    )

    eval_metrics = parse_eval_config(config["eval"])

    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate,
        shuffle=True,
        batch_size=config["train"]["heatmaps"]["batch_size"],
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate,
        batch_size=config["train"]["heatmaps"]["batch_size"],
        num_workers=workers,
    )
    
    if test_dataset is None:
        test_loader = val_loader
    else:
        test_loader = DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate,
            batch_size=config["train"]["heatmaps"]["batch_size"],
            num_workers=workers,
        )

    heatmap_trainer = HeatmapTrainer(
        detector=detector,
        epoch_num=config["train"]["heatmaps"]["num_epochs"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        save_dir=model_dir,
        save_best=config["train"]["heatmaps"]["save_best"],
        best_checkpoints_from=config["train"]["heatmaps"]["best_checkpoints_from"],
        device=device,
        overwrite=overwrite,
        checkpoint_every=config["train"]["heatmaps"]["checkpoint_every"],
        config=config,
        jupyter_mode=jupyter_mode,
        log_every=config["train"]["heatmaps"]["log_every"],
    )

    extractor_trainer = KeypointsExtractorTrainer(
        detector,
        val_dataset,
        model_dir,
        checkpoint_type=config["train"]["extractor"]["checkpoint_type"],
        num_iters=config["train"]["extractor"]["num_iterations"],
        similarity_scale=config["train"]["extractor"]["similarity_scale"],
        sim_thresh=config["train"]["extractor"]["sim_thresh"],
        params_bounds=config["train"]["extractor"]["params_bounds"],
        num_init_keypoints=config["train"]["extractor"]["num_init_points"],
        jupyter_mode=jupyter_mode,
    )

    evaluator = Evaluator(
        detector,
        eval_metrics,
        test_loader,
        model_dir,
        checkpoints=config["eval"]["checkpoints"],
        device=device,
    )

    heatmap_trainer.train()
    extractor_trainer.train()
    evaluator.eval()


def compose_split_datasets(config):

    if "is_precomp" in config["data"]["split_info"]:
        precomputed = config["data"]["split_info"]["is_precomp"]
    else:
        precomputed = False

    if precomputed:
        split_check = compose_split_check(config["data"], config["data"]["split_info"], precomp=precomputed)
        if split_check != config["data"]["split_info"]:
            diff = DeepDiff(split_check, config["data"]["split_info"], verbose_level=1)["values_changed"]
            diff_str = (
                str(diff)
                .replace("new_value", "kfold_info")
                .replace("old_value", "kfold_check")
            )
            logging.error(diff_str)
            raise ValueError("Data check not passed in precomputed dataset.")

    if config["data"]["split_info"]["split_type"] == "kfold":
        subdirs = ["fold_%i" % x for x in range(config["data"]["split_info"]["num_folds"])]
    else:
        subdirs = ["train", "valid"]

    datasets = {}
    for subdir_name in subdirs:
        if precomputed:
            subdir_yaml_path = os.path.join(config["data"]["dataset_dir"], subdir_name, "precomp_data.yaml")
            dataset = PrecomputedDataset(subdir_yaml_path)
        else:
            subdir_yaml_path = os.path.join(
                config["data"]["dataset_dir"], subdir_name, ".".join([subdir_name, "yaml"])
            )
            dataset = get_heatmaps_dataset(subdir_yaml_path, config)
        datasets[subdir_name] = dataset

    return datasets


def aggregate_kfold_results(config, kfold_info, model_dir):

    checkpoints = config["eval"]["checkpoints"]
    meters = parse_eval_config(config["eval"])
    meters_names = ["_".join([meter.name_group, meter.name]) for meter in meters]

    kfold_results = {y: {x: [] for x in meters_names} for y in checkpoints}

    for fold_i in range(kfold_info["num_folds"]):
        fold_name = "_".join(["fold", str(fold_i)])
        fold_dir = os.path.join(model_dir, fold_name)
        results = load_yaml(os.path.join(model_dir, fold_name, "results.yml"))

        for checkpoint_name in checkpoints:
            for meter_name in meters_names:
                kfold_results[checkpoint_name][meter_name] = results[checkpoint_name][
                    meter_name
                ]

    if config["eval"]["fold_aggregation"] == "mean":
        fold_aggregator = np.mean
    elif config["eval"]["fold_aggregation"] == "max":
        fold_aggregator = np.max
    else:
        fold_aggregator = np.min

    for checkpoint_name in checkpoints:
        for meter_name in meters_names:
            kfold_results[checkpoint_name][meter_name] = float(
                fold_aggregator(kfold_results[checkpoint_name][meter_name])
            )

    write_yaml(os.path.join(model_dir, "kfold_results.yml"), kfold_results)


def train_kfold(
    config,
    kfold_info,
    model_dir,
    kfold_dir,
    workers,
    device,
    overwrite,
    jupyter_mode=False,
):

    makedir_overwrite(model_dir, overwrite)
    datasets = compose_split_datasets(config, kfold_info, kfold_dir)

    for fold_i in range(kfold_info["num_folds"]):

        test_dataset = datasets["fold_%i" % fold_i]
        tqdm.write(
            "\nFold %i tranining.\n------------------------------------------" % fold_i
        )
        test_dataset.disable_augs()
        not_test_datasets = [
            datasets["fold_%i" % x]
            for x in range(kfold_info["num_folds"])
            if x != fold_i
        ]

        for ds in not_test_datasets:
            ds.enable_augs()
        trainval_dataset = ConcatCollaterDataset(not_test_datasets)
        train_ids, val_ids = split_trainval_ids(
            trainval_dataset, 
            len(config["model"]["class_labels_map"].values()), 
            config["train"]["train_size"], 
            kfold_info["pseudostrat"]
        )
        train_dataset = SubsetCollater(trainval_dataset, train_ids)
        val_dataset = SubsetCollater(trainval_dataset, val_ids)
        
        fold_name = "_".join(["fold", str(fold_i)])
        fold_out_dir = os.path.join(model_dir, fold_name)

        single_trainval_run(
            config,
            train_dataset,
            val_dataset,
            device,
            workers,
            fold_out_dir,
            overwrite=False,
            test_dataset=test_dataset,
            jupyter_mode=jupyter_mode,
        )

    aggregate_kfold_results(config, kfold_info, model_dir)


def train_trainval(
    config,
    split_info,
    model_dir,
    split_dir,
    workers,
    device,
    overwrite,
    jupyter_mode=False,
):

    datasets = compose_split_datasets(config, split_info, split_dir)
    datasets["train"].enable_augs()
    datasets["val"].disable_augs()
    single_trainval_run(
        config,
        datasets["train"],
        datasets["val"],
        device,
        workers,
        model_dir,
        overwrite=overwrite,
        jupyter_mode=jupyter_mode,
    )


def precompute(
    config,
    split_info,
    input_dir,
    output_dir,
    workers,
    storing_method,
    overwrite,
    jupyter_mode=False,
):
    

    makedir_overwrite(output_dir, overwrite=overwrite)

    if split_info["split_type"] == "kfold":
        subdirs_names = ["fold_%i" % x for x in range(split_info["num_folds"])]
        no_augs_list = [True] * len(subdirs_names)
        num_precomps_list = [config["train"]["heatmaps"]["num_precomputions"]] * len(
            subdirs_names
        )
    elif split_info["split_type"] == "trainval":
        subdirs_names = ["train", "val"]
        no_augs_list = [False, True]
        num_precomps_list = [config["train"]["heatmaps"]["num_precomputions"], 0]
    else:
        raise ValueError("Unknown split type: %s" % split_info["split_type"])

    for subdir_name, add_no_augs, num_precomps in zip(
        subdirs_names, no_augs_list, num_precomps_list
    ):

        tqdm.write("\nProcessing %s..." % subdir_name)

        dataset = get_heatmaps_dataset(
            os.path.join(input_dir, subdir_name, ".".join([subdir_name, "yaml"])),
            config,
        )
        check_data = {"num_samples": len(dataset)}
        out_dir = os.path.join(output_dir, subdir_name)
        precomp = Precompution(
            dataset,
            out_dir,
            repeats_num=num_precomps,
            num_workers=workers,
            check_data=check_data,
            add_sample_with_no_aug=add_no_augs,
            overwrite=overwrite,
            jupyter_mode=jupyter_mode
        )
        precomp.make()

    split_info_new = split_info
    split_info_new["precomp"] = True
    split_info_new["storing_method"] = storing_method

    split_dict = compose_split_check(config, split_info_new, precomp=True)

    write_yaml(os.path.join(output_dir, "split_info.yml"), split_dict)
    tqdm.write("Done!")
