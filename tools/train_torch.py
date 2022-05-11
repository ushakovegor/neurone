import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import EngineCheckpointerCallback
from animus.torch.engine import CPUEngine, DDPEngine, DPEngine, GPUEngine, XLAEngine
from neurone.runners.detection import Experiment


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
        "--model_dir", type=str, help="directory to save the model", required=True)

    parser.add_argument(
        "--dataset_dir", type=str, help="path to train dataset yaml", required=True
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
        default=0,
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = get_config(args.config)
    runner = Experiment(num_epochs=15, engine=E2E[args.engine](fp16=args.fp16))
    runner.run()

if __name__ == "__main__":
    main()
