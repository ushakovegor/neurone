import numpy as np
from nucleidet.detectors.base import BaseDetector
from nucleidet.models.heatmap.dummy import DummyModel, RandomModel


class DummyDetector(BaseDetector, DummyModel):
    """
    Dummy nuclei detector -- detects a single nuclei in the predefined postions

    Parameters
    ----------
    num_classes: int
        number of keypoints classes
    keypoints: endoanalysis.targets.Keypoints
        kepoints for a prediefined answer. If not provided, a single keypoint
        at the center of the image will be provided py  detect_single method
    confidences: ndarrau of float
        confidences of the keypoints
    """

    def __init__(self, num_classes, keypoints=None, confidences=None):
        DummyModel.__init__(self, num_classes, keypoints, return_torch=False)
        self.memorized_confidences = confidences

    def detect_single(self, image):

        predicted_kp = super().fetch_keypoints(image.shape[2], image.shape[1])

        if self.memorized_confidences is None:
            confidences = np.ones(len(predicted_kp))
        else:
            confidences = self.memorized_confidences

        return predicted_kp, confidences


class RandomDetector(BaseDetector, RandomModel):
    """
    Random nuclei detector. It puts keypoints at random positions for each image.

    Parameters
    ----------
        num_classes: int
        number of keypoints classes
    num_keypoints: int or tuple of size 2
        if int, the number of random keypoints to "detect" on each image.
        if tuple, the range of this number
    """

    def __init__(self, num_classes=1, num_keypoints=None):

        RandomModel.__init__(
            self,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            return_torch=False,
        )

    def detect_single(self, image):

        keypoints = super().fetch_keypoints(image.shape[2], image.shape[1])
        num_keypoints = len(keypoints)
        confidences = np.random.uniform(0, 1, num_keypoints)

        return keypoints, confidences
