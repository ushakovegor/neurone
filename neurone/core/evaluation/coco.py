import os
import numpy as np
import shutil
import json
from random import randint
from endoanalysis.targets import Keypoints, KeypointsBatch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOevalMoreKP(COCOeval):
    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(
                0, arUpdatedeaRng="small", maxDets=self.params.maxDets[2]
            )
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=500)
            stats[1] = _summarize(1, maxDets=500, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=500, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=500, areaRng="medium")
            stats[4] = _summarize(1, maxDets=500, areaRng="large")
            stats[5] = _summarize(0, maxDets=500)
            stats[6] = _summarize(0, maxDets=500, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=500, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=500, areaRng="medium")
            stats[9] = _summarize(0, maxDets=500, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()


class COCOmAP:
    """
    COCO mAP evaluator.
    """

    def __init__(
        self,
        keypointsGt,
        keypointsDt,
        confidences,
        area=1,
        scale=1,
        temp_dir="./tmp_coco",
        overwrite=False,
    ):
        assert keypointsGt.shape[-1] == keypointsDt.shape[-1]
        if type(keypointsGt) == KeypointsBatch and type(keypointsDt) == KeypointsBatch:
            self.batch = True
        elif type(keypointsGt) == Keypoints and type(keypointsDt) == Keypoints:
            self.batch = False
        else:
            raise Exception("Pls check your types of keypoints.")
        self.temp_dir = temp_dir
        self.area = area
        self.scale = scale
        self.overwrite = overwrite
        self.ids = set()
        self.keypointsGt = self.coco_transform(keypointsGt)
        self.keypointsDt = self.coco_transform(keypointsDt, confidences)

    def coco_transform(self, keypoints, confidences=None):
        anno_name = "annotation"
        coco_keypoints = {"images": [], "annotations": [], "categories": []}
        categories = set()
        images = set()
        # if confidences is None:
        #     confidences = np.ones(keypoints.shape[0])
        for k, keypoint in enumerate(keypoints):
            annotation = {}
            annotation["num_keypoints"] = 1
            annotation["keypoints"] = [0 for i in range(3)]
            # keypoints_list = keypoint.strip().split(" ")
            # keypoints_list = list(int(float(x)) for x in keypoints_list)
            if self.batch:
                annotation["image_id"] = int(keypoint[0])
                annotation["keypoints"][0:2] = keypoint[1:3].tolist()
                annotation["keypoints"][2] = 1
                annotation["category_id"] = int(keypoint[3])
                # annotation['category_id'] = 0
            else:
                annotation["image_id"] = 0
                annotation["keypoints"][0:2] = keypoint[:2].tolist()
                annotation["keypoints"][2] = 1
                annotation["category_id"] = int(keypoint[2])
                annotation["category_id"] = 0

            if annotation["image_id"] not in images:
                images.add(annotation["image_id"])
                coco_keypoints["images"].append({"id": annotation["image_id"]})
            if confidences is not None:
                annotation["score"] = float(confidences[k])
            annotation["bbox"] = [0, 0, 400, 400]
            id = randint(0, 10 * keypoints.shape[0])
            while id in self.ids:
                id = randint(0, 10 * keypoints.shape[0])
            annotation["id"] = id
            self.ids.add(id)
            annotation["area"] = self.area
            annotation["iscrowd"] = 0
            categories.add(annotation["category_id"])
            coco_keypoints["annotations"].append(annotation)

        for category in categories:
            coco_keypoints["categories"].append({"id": category})
        if os.path.isdir(self.temp_dir):
            if self.overwrite:
                shutil.rmtree(self.temp_dir)
            else:
                raise Exception(
                    "Temp directory is not empty and overwrite flag is disabled, pls set overwrite flag as True."
                )
        os.makedirs(self.temp_dir)
        with open(f"{self.temp_dir}/{anno_name}.json", "w") as anno_file:
            json.dump(coco_keypoints, anno_file)

        coco_anno = COCO(f"{self.temp_dir}/{anno_name}.json")
        shutil.rmtree(self.temp_dir)
        return coco_anno

    def evaluate(self):
        cocoEval = COCOevalMoreKP(self.keypointsGt, self.keypointsDt, "keypoints")
        cocoEval.params.kpt_oks_sigmas = np.array([self.scale])
        cocoEval.params.maxDets = [500]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
