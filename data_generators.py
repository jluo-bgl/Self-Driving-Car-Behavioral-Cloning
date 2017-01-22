import numpy as np
import cv2
import scipy.ndimage
from data_load import FeedingData


def image_itself(feeding_data):
    return feeding_data.image(), feeding_data.steering_angle


def shift_image_generator(angle_offset_pre_pixel=0.003):
    def _generator(feeding_data):
        image, angle, _ = _shift_image(
            feeding_data.image(), feeding_data.steering_angle, 80, 20, angle_offset_pre_pixel=angle_offset_pre_pixel)
        return image, angle

    return _generator


def random_generators(*generators):
    def _generator(feeding_data):
        index = np.random.randint(0, len(generators))
        return generators[index](feeding_data)

    return _generator


def flip_generator(feeding_data):
    image, angle = feeding_data.image(), feeding_data.steering_angle
    return cv2.flip(image, 1), -angle


def pipe_line_generators(*generators):
    """
    pipe line of generators, generator will run one by one
    :param generators:
    :return:
    """
    def _generator(feeding_data):
        intermediary_feeding_data = feeding_data
        for generator in generators:
            image, angle = generator(intermediary_feeding_data)
            intermediary_feeding_data = FeedingData(image, angle)
        return intermediary_feeding_data.image(), intermediary_feeding_data.steering_angle

    return _generator


def pipe_line_random_generators(*generators):
    def _generator(feeding_data):
        count = np.random.randint(0, len(generators))
        intermediary_feeding_data = feeding_data
        for index in range(count):
            generator = generators[index]
            image, angle = generator(intermediary_feeding_data)
            intermediary_feeding_data = FeedingData(image, angle)
        return intermediary_feeding_data.image(), intermediary_feeding_data.steering_angle

    return _generator


def filter_generator(generator, angle_threshold=0.1):
    def _generator(feeding_data):
        image, angle = None, None
        for index in range(20):
            if angle is None or angle <= angle_threshold:
                image, angle = generator(feeding_data)
            else:
                break

        return image, angle

    return _generator


def _shift_image(image, steer, left_right_shift_range, top_bottom_shift_range, angle_offset_pre_pixel=0.003):
    shift_size = round(left_right_shift_range * np.random.uniform(-0.5, 0.5))
    steer_ang = steer + shift_size * angle_offset_pre_pixel
    top_bottom_shift_size = round(top_bottom_shift_range * np.random.uniform(-0.5, 0.5))
    image_tr = scipy.ndimage.interpolation.shift(image, (top_bottom_shift_size, shift_size, 0))
    return image_tr, steer_ang, shift_size


