from threading import Timer
import logging

class TrafficObject(object):

    def set_car_state(self, car_state):
        pass

    @staticmethod
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        #default: if a sign is 10% of the height of frame
        obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        return obj_height / frame_height > min_height_pct

class RedTrafficLight(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed']=0

class GreenTrafficLight(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')

class Person(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        car_state['speed'] = 0

class Box(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        car_state['speed'] = 0

class Tree(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        car_state['speed'] = 0

class Right(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('right: car turning right')
        #Should we include angle variable in set_car_state?
        car_state['angle'] = #90?

class Left(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('left: car turning left')
        #Should we include angle variable in set_car_state?
        car_state['angle'] = #-90?
