import warnings
import numpy as np
from nucleidet.evaluation.metrics import compose_tp_labels
from nucleidet.evaluation.metrics import avg_precisions_from_composed


class CumulativeMetric:
    def __init__(self, name="generic", name_group="generics", precision=6):
        self.name = name
        if name_group is None:
            name_group = name
        self.name_group = name_group
        self.precision = precision

    def reset():
        raise NotImplementedError

    def update():
        raise NotImplementedError

    def compute():
        raise NotImplementedError

    def get_logging_info(self):
        return self.name, self.name_group, self.value, self.precision

    def best_is_max(self):
        return True

    def min_max_factor(self):
        if self.best_is_max():
            return 1
        else:
            return -1

    def get_value(self):
        return self.value


class HeatmapL(CumulativeMetric):
    def __init__(self, degree=2, name="L_metric", name_group="L_mertics"):
        super().__init__(name, name_group)
        self.degree = degree
        self.reset()

    def reset(self):
        self.errors = np.empty(0)
        self.value = 0

    def update(self, batch_gt, batch_pred):

        heatmaps_gt = batch_gt["heatmaps"].cpu().detach().numpy()
        heatmaps_pred = batch_pred["heatmaps"].cpu().detach().numpy()
        new_errors = np.power(np.abs(heatmaps_gt - heatmaps_pred), self.degree).mean(
            axis=(1, 2, 3)
        )
        new_errors = np.power(new_errors, 1 / self.degree)
        self.errors = np.hstack([self.errors, new_errors])

    def compute(self):
        self.value = np.mean(self.errors)

    def best_is_max(self):
        return False

    def __repr__(self):
        return "HeatmapL mertic of degree %i" % self.degree


class Huber(CumulativeMetric):
    def __init__(self, delta=2, name="Huber_metric", name_group="Huber_metrics"):
        super().__init__(name, name_group)
        self.delta = delta
        self.c0 = np.array(0.5)
        self.c1 = np.array(0.5 * delta)
        self.c2 = np.array(0.5 * delta * delta)
        self.reset()

    def reset(self):
        self.mean_hubers = np.empty(0)
        self.value = 0

    def update(self, batch_gt, batch_pred):
        heatmaps_gt = batch_gt["heatmaps"].cpu().detach().numpy()
        heatmaps_pred = batch_pred["heatmaps"].cpu().detach().numpy()
        diffs = np.abs(heatmaps_gt - heatmaps_pred)
        squares = np.multiply(self.c0, np.power(heatmaps_gt - heatmaps_pred, 2))
        linears = np.add(np.multiply(self.c1, diffs), self.c2)
        hubers = np.where(diffs < self.delta, squares, linears)
        means = hubers.mean(axis=(2, 3)).flatten()
        self.mean_hubers = np.hstack([self.mean_hubers, means])

    def compute(self):
        self.value = np.mean(self.mean_hubers)

    def best_is_max(self):
        return False

    def __repr__(self):
        return "Huber loss mertic with delta %f" % self.delta


class mAPmetric(CumulativeMetric):
    def __init__(
        self,
        similarity,
        class_labels,
        sim_thresh=0.5,
        name="mAP",
        name_group="mean_AP_meters",
        return_nans=False,
    ):
        super().__init__(name, name_group)
        self.class_labels = class_labels
        if type(class_labels) is int:
            self.class_labels = [class_labels]
        self.similarity = similarity
        self.sim_thresh = sim_thresh
        self.return_nans = return_nans
        self.reset()

    def reset(self):
        self.confidences = {}
        self.pred_is_tp = {}
        self.num_gt = {}
        self.classes_pred = {}
        self.value = 0

        for class_label in self.class_labels:
            self.confidences[class_label] = np.empty(0)
            self.pred_is_tp[class_label] = np.empty(0)
            self.num_gt[class_label] = 0
            self.classes_pred[class_label] = np.empty(0)

    def update_one_class(self, batch_gt, batch_pred, class_label):

        keypoints_gt = batch_gt["keypoints"]
        keypoints_pred = batch_pred["keypoints"]
        confidences = batch_pred["confidences"]

        keypoints_gt = keypoints_gt[keypoints_gt.classes() == class_label]
        confidences = confidences[keypoints_pred.classes() == class_label]
        keypoints_pred = keypoints_pred[keypoints_pred.classes() == class_label]

        confidences, pred_is_tp = compose_tp_labels(
            self.similarity,
            self.sim_thresh,
            "all",
            keypoints_gt,
            keypoints_pred,
            confidences,
        )

        self.pred_is_tp[class_label] = np.hstack(
            [self.pred_is_tp[class_label], pred_is_tp]
        )
        self.confidences[class_label] = np.hstack(
            [self.confidences[class_label], confidences]
        )
        self.num_gt[class_label] += len(keypoints_gt)

    def update(self, batch_gt, batch_pred):
        for class_label in self.class_labels:
            self.update_one_class(batch_gt, batch_pred, class_label)

    def compute_avg_precs_one_class(self, class_label):

        classes_pred = np.ones_like(self.pred_is_tp[class_label]) * class_label
        classes_gt = np.ones(self.num_gt[class_label]) * class_label

        avg_precisions, _ = avg_precisions_from_composed(
            self.confidences[class_label],
            self.pred_is_tp[class_label],
            classes_gt,
            classes_pred,
            [class_label],
        )

        return avg_precisions[class_label]

    def compute(self):
        mAP = 0
        num_not_none_classes = len(self.class_labels)
        for class_label in self.class_labels:
            AP = self.compute_avg_precs_one_class(class_label)
            if not AP is None:
                mAP += self.compute_avg_precs_one_class(class_label)
            else:
                num_not_none_classes -= 1

        if num_not_none_classes != 0:
            mAP /= num_not_none_classes
            self.value = float(mAP)
        else:
            if self.return_nans:
                self.value = None
            else:
                self.value = 0
            warnings.warn("No objects to match! for %s" % str(self))

    def best_is_max(self):
        return True

    def __repr__(self):

        return " ".join(
            ["APmeter for the classses:"] + [str(x) for x in self.classes_pred]
        )
