#
import torch, torchvision
#
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
#
import numpy as np
import cv2
#



class maskrcnn_inf:
    def __init__(self, thresh=0.5, model='50FPN', cpu= True):
        self.config = get_cfg()

        # RUN on CPU
        if cpu:
            cfg.MODEL.DEVICE = 'cpu'
        
        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        if model == '50FPN':
            self.config.merge_from_file(model_zoo.get_config_file(
                ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")))
            self.config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(self.config)

    def predict(self, img, show=False):
        
        
        output = self.predictor(img)

        # post processing

        if show:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.config.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(output["instances"].to("cpu"))
            return out.get_image()[:, :, ::-1]

        return output['instances'].to('cpu')
