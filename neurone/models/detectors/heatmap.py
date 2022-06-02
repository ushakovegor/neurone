import numpy as np
import torch
from torch import nn
from endoanalysis.targets import KeypointsBatch, keypoints_list_to_batch
from endoanalysis.similarity import KPSimilarity
from neurone.models.detectors.base import BaseDetector
from neurone.data.keypoints import rescale_keypoints


class ClassSeparator(nn.Module):
    def forward(self, x):
        raise NotImplementedError()


class TrivialClassSeparator(ClassSeparator):
    def forward(self, x):
        return x


class SoftmaxClassSeparator(ClassSeparator):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class MaxClassSeparator(ClassSeparator):
    def __init__(self):
        super(MaxClassSeparator, self).__init__()
        self.epsilon = nn.Parameter(torch.tensor(1e-5), requires_grad=False)
        # self.register_parameter(self.epsilon)

    def forward(self, x):
        maxes = torch.max(x, axis=1, keepdim=True).values
        maxes = torch.where(maxes > 0, maxes, self.epsilon)
        return x / maxes


class KeypointsExtractor(nn.Module):
    def __init__(
        self, min_peak_value, pooling_scale, out_image_shape, supression_range
    ):
        super(KeypointsExtractor, self).__init__()
        self.out_image_shape = out_image_shape
        self.sim_thresh = 0.5
        self._peak_lower_bond = 0.01
        self._peak_greater_bond = 0.99
        self.set_params(min_peak_value, pooling_scale, supression_range)

    def set_params(self, min_peak_value, pooling_scale, supression_range):
        min_peak_value = np.max([min_peak_value, self._peak_lower_bond])
        self.min_peak_value = np.min([min_peak_value, self._peak_greater_bond])
        self.confidence_factor = 1.0 / (1.0 - self.min_peak_value)
        self.similarity = KPSimilarity(scale=supression_range)

        padding = np.floor(pooling_scale / 2).astype(int)
        self.pooler = nn.MaxPool2d(kernel_size=pooling_scale, padding=padding, stride=1)

    def keypoints_from_peaks_crossclass(self, heatmaps):
        num_classes = heatmaps.shape[1]
        heatmaps_cumulative, max_ids = heatmaps.max(axis=1, keepdim=False)
        pooled = self.pooler(heatmaps_cumulative)
        rel_maxs = pooled == heatmaps_cumulative
        image_ids, y_coords, x_coords = torch.where(rel_maxs)
        class_ids = max_ids[image_ids, y_coords, x_coords]
        peak_values = heatmaps[image_ids, class_ids, y_coords, x_coords]
        keypoints = (
            torch.vstack([image_ids, x_coords, y_coords, class_ids])
            .cpu()
            .numpy()
            .T.astype(float)
        )

        keypoints = KeypointsBatch(keypoints)

        return keypoints, peak_values.detach().cpu().numpy()

    def keypoints_from_peaks(self, heatmaps):
        """
        Extract keypoints from the heatmaps using max pooling.

        Parameters
        ----------
        heatmaps: ndarray
            images heatmaps, the size is (n_imgs, num_classes, y_size, x_size)

        Returns
        -------
        keypoints: endoanalysis.targets.KeypointsBatch
            extracted keypoints
        scores: torch.tensor
            the scores of the extracted keypoints
        """

        pooled = self.pooler(heatmaps)
        rel_maxs = pooled == heatmaps
        image_ids, class_ids, y_coords, x_coords = torch.where(rel_maxs)
        peak_values = heatmaps[image_ids, class_ids, y_coords, x_coords]
        keypoints = (
            torch.vstack([image_ids, x_coords, y_coords, class_ids])
            .cpu()
            .numpy()
            .T.astype(float)
        )

        keypoints = KeypointsBatch(keypoints)

        return keypoints, peak_values.detach().cpu().numpy()

    def supress(self, keypoints_batch, confidences_batch, batch_size):

        if not len(keypoints_batch):
            return keypoints_batch, confidences_batch

        keypoints_batch_supressed = []
        confidences_batch_supressed = []
        for image_i in range(batch_size):
            keypoints = keypoints_batch.from_image(image_i)

            confidences = confidences_batch[keypoints_batch.image_labels() == image_i]
            confidences_ids = np.argsort(confidences)
            kp_sorted = keypoints[confidences_ids][::-1]

            sim_matrix = self.similarity.matrix(kp_sorted, kp_sorted)
            overlap_matrix = sim_matrix > self.sim_thresh
            ids_to_keep = np.sum(np.triu(overlap_matrix, 1), axis=0) == 0
            keypoints_batch_supressed.append(kp_sorted[ids_to_keep])
            confidences_batch_supressed.append(
                confidences[confidences_ids][ids_to_keep]
            )

        keypoints_batch_supressed = keypoints_list_to_batch(keypoints_batch_supressed)
        confidences_batch_supressed = np.concatenate(confidences_batch_supressed)

        return keypoints_batch_supressed, confidences_batch_supressed

    def normalize_confidences(self, confidences):
        confidences -= self.min_peak_value
        confidences *= self.confidence_factor
        confidences[confidences > 1.0] = 1.0
        confidences[confidences < 0.0] = 0.0
        return confidences

    def forward(self, heatmaps):
        batch_size = heatmaps.shape[0]
        keypoints, peak_values = self.keypoints_from_peaks_crossclass(heatmaps)

        valid_peak_ids = peak_values > self.min_peak_value
        keypoints = keypoints[valid_peak_ids]
        confidences = peak_values[valid_peak_ids]
        confidences = self.normalize_confidences(confidences)
        keypoints, confidences = self.supress(keypoints, confidences, batch_size)

        keypoints = rescale_keypoints(
            keypoints,
            in_image_shape=heatmaps.shape[2:4],
            out_image_shape=self.out_image_shape,
        )

        return keypoints, confidences


class HeatmapDetector(nn.Module, BaseDetector):
    """
    Nuclei detector based on heatmap model.
    A model converts images into several heatmaps (one heatmap per class),
    from which the keypoints are eextracted

    Parameters
    ----------
    preprocessor: nucleidet.data.preprocess.Preprocessor
        preprocessor instance wich converts images to the form requird byt the model
    heatmap_model: nucleidet.models.heatmap.HeatmapModel
        torch model to extract heatmaps from the images
    class_separator: nucleidet.detectors.heatmap.ClassSeparator
        class separator to force the keypoints from differen classes be present at
        the same spots.
    """

    def __init__(
        self, prerprocessor, class_separator, heatmap_model, keypoints_extractor
    ):
        super(HeatmapDetector, self).__init__()
        self.preprocessor = prerprocessor
        self.heatmap_model = heatmap_model
        self.class_separator = class_separator
        self.keypoints_extractor = keypoints_extractor
        self.set_predict_keypoints(True)

    def set_predict_keypoints(self, flag):
        if flag:
            self.predict_keypoints = True
        else:
            self.predict_keypoints = False

    def topk(self, heatmaps, k=1):
        """
        Extract keypoints as topk values.

        Parameters
        ----------
        heatmaps: ndarray
            images heatmaps, the size is (n_imgs, num_classes, y_size, x_size)

        Returns
        -------
        keypoints: endoanalysis.targets.KeypointsBatch
            extracted keypoints
        scores: torch.tensor
            the scores of the extracted keypoints

        Note
        ----
        After this function a supression algorithm is usually used.
        """

        num_images, _, y_size, x_size = heatmaps.shape
        topk = torch.topk(heatmaps.flatten(start_dim=1, end_dim=3), k=k, dim=-1)
        topk_inds = topk.indices
        scores = topk.values

        classes = torch.floor(topk_inds / (y_size * x_size))
        topk_inds = topk_inds - classes * y_size * x_size
        y_coords = torch.floor((topk_inds) / x_size)
        topk_inds = topk_inds - y_coords * x_size
        x_coords = torch.floor((topk_inds))

        keypoints_pred = np.stack(
            [x_coords.numpy(), y_coords.numpy(), classes.numpy()], -1
        )
        images_ids = np.arange(num_images).repeat(k).reshape(-1, 1)
        keypoints_pred = np.hstack(
            [images_ids, keypoints_pred.reshape(num_images * k, -1)]
        )
        keypoints_pred = KeypointsBatch(keypoints_pred)

        return keypoints_pred, scores.reshape(-1)

    def detect_multi(self, images):
        _, keypoints, confidences = self(images)
        return keypoints, confidences

    def forward(self, images):
        images = self.preprocessor(images)
        heatmaps = self.heatmap_model(images)

        # heatmaps = self.class_separator(heatmaps)

        if self.predict_keypoints:
            keypoints, confidences = self.keypoints_extractor(heatmaps)
        else:
            keypoints = None
            confidences = None

        return heatmaps, keypoints, confidences
