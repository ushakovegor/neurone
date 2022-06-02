import numpy as np
from endoanalysis.targets import keypoints_list_to_batch


class BaseDetector:
    """
    Base class for nuclei detectors.
    """

    def __init__(self):
        pass

    def detect_single(self, image):
        """
        Detect nuclei for single image

        Paramteres
        ----------
        image: ndarray
            input image, the shape is (C, H, W)

        Returns
        -------
        keypoints: ndarray
            detected keypoints
        confidences: ndarray
            confidences of the predictions
        """

        raise NotImplementedError()

    def detect_multi(self, images):
        """
        Detect nuclei for multiple images.

        Paramteres
        ----------
        images: iterable of ndarray
            input images, the shape of each image is (C, H, W)

        Returns
        -------
        images_keypoints: endoanalysis.targets.Keypoints
            detected keypoints for all images.
        images_confidences: ndarray
            concatenated confidences of the predictions

        """
        images_keypoints = []
        images_confidences = []
        for im in images:
            keypoints, confidences = self.detect_single(im)
            images_keypoints.append(keypoints)
            images_confidences.append(confidences)

        return keypoints_list_to_batch(images_keypoints), np.concatenate(
            images_confidences
        )
