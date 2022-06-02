import torch


class BasicPreprocessor:
    """
    Basic class for image prerprocessor.
    """

    def __call__(self, images):
        """
        Preprocess batch of images.

        Parameters
        ----------
        images: ndarray like
            batch of images, the shape is (N, C, H, W)

        Returns
        ------
            preprocessed batch, the shape is (N, C, H, W)
        """
        return images


class Normalizator:
    """
    Basic class for image prerprocessor.

    Parameters
    ----------
    channels_mean: tuple of float
        mean channels values
    channels_std: tuple of int
        channels standart deviations
    """

    def __init__(self, channels_mean, channels_std):
        self.channels_mean = channels_mean
        self.channels_std = channels_std

    def __call__(self, images):
        """
        Preprocess batch of images.

        Parameters
        ----------
        images: torch.tensor
            batch of images, the shape is (N, C, H, W)

        Returns
        ------
            preprocessed batch, the shape is (N, C, H, W)
        """

        images -= torch.tensor(self.channels_mean).reshape(1, -1, 1, 1)
        images /= torch.tensor(self.channels_std).reshape(1, -1, 1, 1)

        return images
