import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData
from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator
from data_generators import center_image_generator, \
    center_left_right_image_generator, \
    left_image_generator, \
    right_image_generator, shift_image_generator, random_generators, pipe_line_generators
from visualization import Video


class TestVideos(unittest.TestCase):
    def test_gif_for_generator(self):
        dataset = DriveDataSet("../datasets/udacity-sample-track-1/driving_log.csv")

        generator = pipe_line_generators(
            center_image_generator,
            shift_image_generator
        )
        Video.from_generators("resources/shift_center_images.gif", dataset[60], 0.0, 20, generator)

        generator = pipe_line_generators(
            left_image_generator,
            shift_image_generator
        )
        Video.from_generators("resources/shift_left_images.gif", dataset[60], 0.2, 20, generator)

        generator = pipe_line_generators(
            right_image_generator,
            shift_image_generator
        )
        Video.from_generators("resources/shift_right_images.gif", dataset[60], -0.2, 20, generator)



