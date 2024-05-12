import cv2, os
from time import perf_counter
import numpy as np
from typing import ClassVar, Dict
from detectron2.config import get_cfg
from detectron2.structures.instances import Instances
from detectron2.engine.defaults import DefaultPredictor
from DensePose.densepose import add_densepose_config
from DensePose.densepose.vis.base import CompoundVisualizer
from DensePose.densepose.vis.extractor import CompoundExtractor, create_extractor
from DensePose.densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
)
class Dense:
    def __init__(self, video, output_path):
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(
            "/content/Features_Extractor_HGR/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        self.predictor = DefaultPredictor(self.cfg)
        self.VISUALIZERS: ClassVar[Dict[str, object]] = {
            "dp_contour": DensePoseResultsContourVisualizer,
        }

        self.vis_specs = ['dp_contour']
        self.visualizers = []
        self.extractors = []
        for self.vis_spec in self.vis_specs:
            self.vis = self.VISUALIZERS[self.vis_spec]()
            self.visualizers.append(self.vis)
            self.extractor = create_extractor(self.vis)
            self.extractors.append(self.extractor)
        self.visualizer = CompoundVisualizer(self.visualizers)
        self.extractor = CompoundExtractor(self.extractors)

        self.context = {
            "extractor": self.extractor,
            "visualizer": self.visualizer
        }

        self.visualizer = self.context["visualizer"]
        self.extractor = self.context["extractor"]
        self.cap = cv2.VideoCapture(video)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_path = '/content/Features_Extractor_HGR/video-outputs/' + output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.outvid_2 = cv2.VideoWriter(self.output_path + '/mask.avi',
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (self.frame_width, self.frame_height))

    def videoWriterMask(self, blackFrame):
        self.outvid_2.write(blackFrame)

    def extractorMask(self, frame):
        bl = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        outputs = self.predictor(frame)['instances']
        data = self.extractor(outputs)
        black_frame = self.visualizer.visualize(bl, data)
        black_frame = cv2.resize(black_frame, (self.frame_width, self.frame_height))
        self.videoWriterMask(black_frame)

    def videoMask(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.extractorMask(frame)
            else:
                break
        self.cap.release()
        self.outvid_2.release()
