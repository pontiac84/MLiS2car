import cv2
import logging
import datetime
import time
import edgetpu.detection.engine
from PIL import Image
from traffic_objects import *

_SHOW_IMAGE = False


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=1, #should this be 1 or 35? or not included at all?
                 #replace with our files
                 model='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_signs_quantized_edgetpu.tflite',
                 label='/home/pi/DeepPiCar/models/object_detection/data/model_result/road_sign_labels.txt',
                 width=640,
                 height=480):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info('Creating a ObjectsOnRoadProcessor...')
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

        # initial edge TPU engine
        logging.info('Initialize Edge TPU with model %s...' % model)
        self.engine = edgetpu.detection.engine.DetectionEngine(model)
        self.min_confidence = 0.30
        self.num_of_objects = 3
        logging.info('Initialize Edge TPU with model done.')

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms


        self.traffic_objects = {0: GreenTrafficLight(),
                                1: Person(),
                                2: RedTrafficLight(),
                                3: Box(),
                                4: Tree(),
                                5: LeftSign(),
                                6: RightSign()}

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug('Processing objects.................................')
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug('Processing objects END..............................')

        return final_frame

    def detect_objects(self, frame):
        logging.debug('Detecting objects...')

        # call tpu for inference
        start_ms = time.time()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_RGB)
        objects = self.engine.DetectWithImage(img_pil, threshold=self.min_confidence, keep_aspect_ratio=True,
                                         relative_coord=False, top_k=self.num_of_objects)
        if objects:
            for obj in objects:
                height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
                width = obj.bounding_box[1][0]-obj.bounding_box[0][0]
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" % (self.labels[obj.label_id], obj.score * 100, width, height))
                box = obj.bounding_box
                coord_top_left = (int(box[0][0]), int(box[0][1]))
                coord_bottom_right = (int(box[1][0]), int(box[1][1]))
                cv2.rectangle(frame, coord_top_left, coord_bottom_right, self.boxColor, self.boxLineWidth)
                annotate_text = "%s %.0f%%" % (self.labels[obj.label_id], obj.score * 100)
                coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                cv2.putText(frame, annotate_text, coord_top_left, self.font, self.fontScale, self.boxColor, self.lineType)
        else:
            logging.debug('No object detected')

        elapsed_ms = time.time() - start_ms

        annotate_summary = "%.1f FPS" % (1.0/elapsed_ms)
        logging.debug(annotate_summary)
        cv2.putText(frame, annotate_summary, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)
        #cv2.imshow('Detected Objects', frame)

        return objects, frame


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)

