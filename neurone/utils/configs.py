import albumentations as A
import cv2
import torch
import json
import segmentation_models_pytorch as smp
from endoanalysis.similarity import KPSimilarity
from neurone.core.train.meters import mAPmetric, HeatmapL, Huber
from neurone.core.train.schedulers import PlateauReducer
from neurone.data.preprocess import BasicPreprocessor
from neurone.models.detectors.heatmap import HeatmapDetector, KeypointsExtractor
from neurone.models.detectors.heatmap import TrivialClassSeparator
from neurone.models.detectors.heatmap import MaxClassSeparator
from neurone.models.detectors.heatmap import SoftmaxClassSeparator
from neurone.core.train.criterions import HeatmapMAE, HeatmapMSE, HeatmapHuber


def get_config(config_path):
    config = json.load(config_path)
    return config


def define_metric(metric, train_val):

    if metric["type"] == "MAE":
        return HeatmapL(degree=1, name="_".join([train_val, "MAE"]), name_group="MAE")
    elif metric["type"] == "MSE":
        return HeatmapL(degree=2, name="_".join([train_val, "MSE"]), name_group="MSE")
    elif metric["type"] == "Huber":
        return Huber(
            delta=metric["delta"],
            name="_".join([train_val, "Huber"]),
            name_group="Hubers",
        )
    elif metric["type"] == "AP":
        similarity = KPSimilarity(metric["similarity"])
        return mAPmetric(
            similarity=similarity,
            class_label=metric["class"],
            sim_thresh=metric["thresh"],
            name="_".join([train_val, "AP"]),
            name_group="_".join(
                ["AP", "class", str(metric["class"]), "sim", str(metric["similarity"])]
            ),
        )
    else:
        raise Exception("Unknown metric: %s" % metric["type"])


def define_optimizer(optimizer_config, model):
    if optimizer_config["type"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config["kwargs"])
    elif optimizer_config["type"] == "Adam":

        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config["kwargs"])
    else:
        raise Exception("Unknown optimizer: %s" % optimizer_config["type"])

    return optimizer


def define_criterion(criterion_config):
    if criterion_config["type"] == "MAE":
        criterion = HeatmapMAE(
            class_weights=criterion_config["class_weights"],
            normalize_weights=criterion_config["normalize_weights"],
        )
    elif criterion_config["type"] == "MSE":
        criterion = HeatmapMSE(
            class_weights=criterion_config["class_weights"],
            normalize_weights=criterion_config["normalize_weights"],
        )
    elif criterion_config["type"] == "Huber":
        criterion = HeatmapHuber(
            class_weights=criterion_config["class_weights"],
            normalize_weights=criterion_config["normalize_weights"],
            delta=criterion_config["delta"],
        )
    else:
        raise Exception("Unknown criterion: %s" % criterion_config)

    return criterion


def define_scheduler(scheduler_config, optimizer, train_metrics, val_metrics):
    if scheduler_config["type"] == "no_scheduler":
        scheduler = None

    elif scheduler_config["type"] == "reduce_on_plateu":
        if scheduler_config["train_val"] == "train":
            metric_to_trace = train_metrics[scheduler_config["metric_type"]]
        elif scheduler_config["train_val"] == "val":
            metric_to_trace = val_metrics[scheduler_config["metric_type"]]
        else:
            raise Exception(
                "Expected 'train or 'val' in config['scheduler']['train_val'], got %s"
                % scheduler_config["scheduler"]["train_val"]
            )
        scheduler = PlateauReducer(
            metric_to_trace,
            optimizer,
            verbose=True,
            mode="max" if metric_to_trace.best_is_max() else "min",
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            threshold=scheduler_config["threshold"],
        )
    else:
        raise Exception("Unknown scheduler: %s" % scheduler_config["type"])

    return scheduler


def parse_train_config(config, model):
    criterion = define_criterion(config["criterion"])
    optimizer = define_optimizer(config["optimizer"], model)

    train_metric = define_metric(config["train_metric"], "train")

    valid_metric = define_metric(config["valid_metric"], "val")

    scheduler = define_scheduler(
        config["scheduler"], optimizer, train_metric, valid_metric
    )

    return criterion, optimizer, scheduler, train_metric, valid_metric


def parse_eval_config(config):
    eval_meters = None
    if config["metric"]["type"] == "mAP":
        similarity = KPSimilarity(scale=config["metric"]["similarity_scale"])
        eval_meters = mAPmetric(
                similarity,
                class_labels=config["metric"]["class_labels"],
                sim_thresh=config["metric"]["sim_thresh"],
                name_group="mAP",
                # name="_".join(["mAP"] + [str(x) for x in item["class_labels"]]),
                name="_".join([str(x) for x in config["metric"]["class_labels"]]),
            )
    else:
        raise Exception("Unknown meter type in eval: %s" % config["metric"]["type"])

    return eval_meters


def model_from_config(config):

    if (
        len(set(config["class_labels_map"].values()))
        != config["model_kwargs"]["classes"]
    ):
        raise ValueError(
            "Number of classes is not the same as the number of values in the class_labels_map."
        )
    if config["model_type"] == "Unet":
        model_type = smp.Unet
    elif config["model_type"] == "Unet++":
        model_type = smp.UnetPlusPlus
    else:
        raise Exception("Unknown model type: %s" % config["model_type"])

    if config["class_separator"] is None:
        class_separator = TrivialClassSeparator()
    elif config["class_separator"] == "softmax":
        class_separator = SoftmaxClassSeparator()
    elif config["class_separator"] == "max":
        class_separator = MaxClassSeparator()
    else:
        raise Exception("Unknown class separator: %s" % config["class_separator"])

    model = model_type(**config["model_kwargs"])

    keypoints_extractor = KeypointsExtractor(
        min_peak_value=config["min_peak_value"],
        pooling_scale=config["pool_scale"],
        supression_range=config["supression_range"],
        out_image_shape=config["input_shape"],
    )
    detector = HeatmapDetector(
        prerprocessor=BasicPreprocessor(),
        heatmap_model=model,
        class_separator=class_separator,
        keypoints_extractor=keypoints_extractor,
    )

    return model #FIXME to detector


def define_extrapolation_mode(extrapolation_mode):
    if extrapolation_mode == "BORDER_CONSTANT":
        return cv2.BORDER_CONSTANT
    elif extrapolation_mode == "BORDER_REPLICATE":
        return cv2.BORDER_REPLICATE
    elif extrapolation_mode == "BORDER_WRAP":
        return cv2.BORDER_WRAP
    elif extrapolation_mode == "BORDER_REFLECT":
        return cv2.BORDER_REFLECT
    elif extrapolation_mode == "BORDER_REFLECT_101":
        return cv2.BORDER_REFLECT_101
    else:
        raise Exception(
            "\nExtrapolation mode should be one of the following: \n"
            + "BORDER_CONSTANT, BORDER_REPLICATE,BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101 \n"
            + "Got:\n%s" % extrapolation_mode
        )


def albumentations_from_config(aug_config):

    border_mode = define_extrapolation_mode(aug_config["border_mode"])
    augs_list = [
        A.augmentations.HueSaturationValue(
            p=aug_config["p_hsv"],
            hue_shift_limit=5,
            sat_shift_limit=5,
            val_shift_limit=5,
        ),
        A.augmentations.GaussNoise(
            p=aug_config["p_noise"], var_limit=aug_config["noise_var"]
        ),
        A.augmentations.Rotate(
            limit=aug_config["rotate_angle"],
            p=aug_config["p_rotate"],
            border_mode=border_mode,
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=aug_config["shift_factor"],
            scale_limit=0,
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_shift"],
        ),
        A.augmentations.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=aug_config["scale_factor"],
            rotate_limit=0,
            border_mode=border_mode,
            p=aug_config["p_scale"],
        ),
        A.augmentations.Perspective(
            scale=(0, aug_config["perspective_factor"]),
            p=aug_config["p_perspective"],
            interpolation=border_mode,
        ),
        A.augmentations.HorizontalFlip(p=aug_config["p_flip_hor"]),
        A.augmentations.VerticalFlip(p=aug_config["p_flip_vert"]),
    ]

    return augs_list
