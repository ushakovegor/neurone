import numpy as np
from endoanalysis.targets import Keypoints, KeypointsBatch


def rescale_keypoints(keypoints, in_image_shape, out_image_shape):
    """
    Rescales keypoints coordinates.

    Parameters
    ----------
    keypoints: endoanalysis.targets.keypoints
        keypoints to rescale
    in_image_shape: tuple of int
        initial image shape, the format is (y_size, x_size)
    out_image_shape: tuple of int
        resulting image shape, the format is (y_size, x_size)

    Returns
    -------
    keypoints_rescaled: endoanalysis.targets.keypoints
        rescaled_keypoints
    """

    keypoints_rescaled = keypoints.copy()

    x_coords = keypoints_rescaled.x_coords().astype(float)
    y_coords = keypoints_rescaled.y_coords().astype(float)
    classes = keypoints_rescaled.classes().astype(float)

    x_coords = np.round((x_coords * out_image_shape[1] / in_image_shape[1]))
    y_coords = np.round((y_coords * out_image_shape[0] / in_image_shape[0]))

    if type(keypoints) is Keypoints:
        keypoints_rescaled = Keypoints(np.vstack([x_coords, y_coords, classes]).T)
    elif type(keypoints) is KeypointsBatch:
        image_lables = keypoints_rescaled.image_labels().astype(float)
        keypoints_rescaled = KeypointsBatch(
            np.vstack([image_lables, x_coords, y_coords, classes]).T
        )
    return keypoints_rescaled
