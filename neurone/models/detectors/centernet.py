from matplotlib import image
import numpy as np
import torch
import warnings
import mmcv
import collections
import copy
import cv2
from endoanalysis.targets import Keypoints
from nucleidet.models.base import BaseDetector
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from nucleidet.utils.pipelines import Compose
from endoanalysis.targets import Keypoints, keypoints_list_to_batch 
from nucleidet.utils.utils import replace_ImageToTensor
from nucleidet.utils.keypoint_utils import WBF_fussed

class CenterNetDetector(BaseDetector):
    """
    CenterNet detector
    """

    def __init__(self, model_path, skip_threshold=(0.32, 0.31)):
        self.skip_threshold = skip_threshold
        self.model = torch.load(model_path)
        self.batch = True
        self.cfg = self.model.cfg
        self.device = next(self.model.parameters()).device  # model device
        self.model.eval()

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
        self.batch = False
        return self.detect_multi([image])

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

        if isinstance(images[0], np.ndarray):
            self.cfg = self.cfg.copy()
            # set loading pipeline type
            self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        self.cfg.data.test.pipeline = replace_ImageToTensor(self.cfg.data.test.pipeline)
        test_pipeline = Compose(self.cfg.data.test.pipeline)
        datas = []
        for img in images:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(images))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)

        keypoints = []
        confidences = []
        for image_result in results:
            image_keypoints = []
            image_confidences = []
            for label, pred_keypoints in enumerate(image_result):
                for keypoint in pred_keypoints:
                    image_keypoints.append([keypoint[0], keypoint[1], label])
                    image_confidences.append(keypoint[4])
            image_keypoints, image_confidences = WBF_fussed(image_keypoints, image_confidences, skip_threshold=self.skip_threshold)
            keypoints.append(Keypoints(image_keypoints))
            confidences.append(image_confidences)
        if not self.batch:
            return keypoints[0], np.array(confidences[0])
        else:
            return keypoints_list_to_batch(keypoints), np.concatenate(confidences)