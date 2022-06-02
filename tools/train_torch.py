import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import EngineCheckpointerCallback
from animus.torch.engine import CPUEngine, DDPEngine, DPEngine, GPUEngine, XLAEngine
from neurone.runners.detection import TestExperiment
from neurone.utils.configs import get_config
from neurone.utils.general import load_yaml


E2E = {
    "cpu": CPUEngine,
    "gpu": GPUEngine,
    "dp": DPEngine,
    "ddp": DDPEngine,
    "xla": XLAEngine,
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("config", help="path to the config file")
    parser.add_argument(
        "--engine", type=str, choices=list(E2E.keys()), required=False, default=CPUEngine)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument(
        "--model_dir", type=str, help="directory to save the model", required=False)

    parser.add_argument(
        "--dataset_dir", type=str, help="path to train dataset yaml", required=False
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="whether to overwrite the model or not",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to load and preprocess the data.",
        default=1,
    )

    args = parser.parse_args()

    return args


def main():
    # args = parse_args()
    with open("/Users/ushakov/projects/neurone/configs/detection.json", "r") as config_file:
        config = json.load(config_file)
    SPLIT_INFO = load_yaml("/Users/ushakov/projects/data/trainval/split_info.yml")
    config["data"]["split_info"] = SPLIT_INFO
    runner = TestExperiment(engine=E2E["cpu"](fp16=True), config=config)
    runner.run()
    print()

if __name__ == "__main__":
    main()
