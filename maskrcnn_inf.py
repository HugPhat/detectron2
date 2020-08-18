#
import torch
import torchvision
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

import matplotlib.colors as mplc


class maskrcnn_inf:
    def __init__(self, thresh=0.5, model='50FPN', cpu=True):
        self.config = get_cfg()

        # RUN on CPU
        if cpu:
            self.config.MODEL.DEVICE = 'cpu'

        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        if model == '50FPN':
            self.config.merge_from_file(model_zoo.get_config_file(
                ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")))
            self.config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(self.config)
        self.metadata = MetadataCatalog.get(self.config.DATASETS.TRAIN[0])

    def toPolygon(self, masks):
      masks = [mask.numpy() for mask in masks]
      result = []
      for i, mask in enumerate(masks):
        mask = (mask * 255).astype('uint8')
        # opencv built-in finding contours methods
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:  # empty mask
          result.append([])
        else:
          contours = [x.flatten() for x in contours]
          contours = [x for x in contours if len(x) >= 6]
          result.append(contours)
      return result

    def cvtAISetFormat(self, data, callback):
      '''
      ** data : [
                  contours coors: list,
                  classes
                ]
      ** callback: function(result) -> modify prefix of API format
      
      function only turn coor to :
      {
        'annotations' : [
          {
            'label' : 'name',
            'coords' : [
              {
                'x' : float,
                'y' : float
              },
              {
                'x' : float,
                'y' : float
              }
            ]
          }
        ]
      }
      '''
      result = {
          'annotations': []
      }

      for (contour, clss) in zip(*data):

        item = {'label': clss, 'coords': []}
        contour = contour[-1]
        for i in range(0, len(contour) - 2, 2):
          item['coords'].append({'x': contour[i + 1], 'y': contour[i]})
        result['annotations'].append(item)

      result = {"result": result}

      if (callback) != None:

        return callback(result)

      return result

    def _create_text_labels(self, classes, class_names):
      """
      Args:
          classes (list[int] or None):
          class_names (list[str] or None):
      Returns:
          list[str] or None
      """
      labels = None
      if classes is not None and class_names is not None and len(class_names) > 0:
          labels = [class_names[i] for i in classes]
      return labels

    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than the color given.
        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.
        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def predict(self, img, callback=None, show=False):

        h, w = img.shape[:2]
        output = self.predictor(img)

        # post processing

        if show:
            v = Visualizer(img[:, :, ::-1],
                           MetadataCatalog.get(self.config.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(output["instances"].to("cpu"))
            return out.get_image()[:, :, ::-1]

        instances = output['instances'].to('cpu')
        masks = instances.pred_masks
        classes = instances.pred_classes
        classes = self._create_text_labels(classes, self.metadata.get("thing_classes", None))
        listCoords = self.toPolygon(masks)

        out = self.cvtAISetFormat([listCoords, classes], callback)
        return out
