import os.path

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import numpy as np
from features_extractors.vectorizer import Vectorizer


class Detectron:
    def __init__(self, video, output_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = 1
        self.predictor = DefaultPredictor(self.cfg)
        self.cap = cv2.VideoCapture(video)
        self.frame_width = 1017
        self.frame_height = 576
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_path = '/home/mcc/PycharmProjects/Features_Extractor_HGR/video-outputs/' + output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.outvid = cv2.VideoWriter(self.output_path,
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (self.frame_width, self.frame_height))
        self.vec = Vectorizer(output_path)

    def videoWriterSkeleton(self, blackFrame):
        self.outvid.write(blackFrame)

    def extractorSkeleton(self, frame):
        h, w = frame.shape[:2]
        bl = np.zeros((h, w, 3), np.uint8)
        outputs = self.predictor(frame)
        v = Visualizer(bl[:, :, ::-1], self.vec, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
        self.videoWriterSkeleton(out.get_image()[:, :, ::-1])

    def videoSkeleton(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.extractorSkeleton(frame)
            else:
                break
        self.cap.release()
        self.outvid.release()
        self.vec.keypoints_csv_generator()
        self.vec.angles_csv_generator()
        self.vec.plotter()

