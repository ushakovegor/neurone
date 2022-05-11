import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import torch
from nucleidet.utils.general import write_yaml


class Evaluator:
    def __init__(
        self,
        detector,
        meters,
        dataloader,
        model_dir,
        checkpoints=["last"],
        device=torch.device("cpu"),
        jupyter_mode=False,
    ):

        self.detector = detector
        self.meters = meters
        self.dataloader = dataloader
        self.model_dir = model_dir
        self.device = device
        self.checkpoints = checkpoints

        if jupyter_mode:
            self.tqdm_class = tqdm_notebook
        else:
            self.tqdm_class = tqdm

    def eval_one_checkpoint(self, checkpoint_type):

        tqdm.write("Evaluating checkpoint ''%s''" % checkpoint_type)
        checkpoint = torch.load(
            os.path.join(
                self.model_dir, "checkpoints/featured/%s.pth" % checkpoint_type
            ),
            map_location=self.device,
        )
        self.detector.to(self.device)
        self.detector.load_state_dict(checkpoint["model_state_dict"])
        for meter in self.meters:
            meter.reset()

        with torch.no_grad():
            for batch in self.tqdm_class(self.dataloader):

                _, keypoints_pred, confidences = self.detector(
                    batch["image"].to(self.device)
                )
                pred_batch = {"keypoints": keypoints_pred, "confidences": confidences}
                for meter in self.meters:
                    meter.update(batch, pred_batch)

        results_checkpoint = {}

        for meter in self.meters:
            meter.compute()
            meter_name_to_save = "_".join([meter.name_group, meter.name])
            results_checkpoint[meter_name_to_save] = meter.get_value()
            tqdm.write(
                "Meter %s value is: %0.4f"
                % (meter_name_to_save, results_checkpoint[meter_name_to_save])
            )
        tqdm.write("----------------------")
        return results_checkpoint

    def eval(self):
        tqdm.write("Evaluation is started.")
        self.detector.eval()

        results = {}
        for checkpoint_type in self.checkpoints:
            results[checkpoint_type] = self.eval_one_checkpoint(checkpoint_type)

        tqdm.write("Evaluation is finished.")
        write_yaml(os.path.join(self.model_dir, "results.yml"), results)
