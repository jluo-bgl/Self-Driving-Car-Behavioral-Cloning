import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData, DriveDataSet
from data_generators import _shift_image, pipe_line_generators, random_generators, flip_generator, filter_generator
from PIL import Image


class TestGeneratorFuncs(unittest.TestCase):
    def test_shift_image(self):
        angle = 1
        image = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        new_image, new_angle, shift_size = _shift_image(image, angle, 3, 0)
        self.assertEqual(new_angle, angle + shift_size * 0.004)
        if shift_size == 1:
            numpy.testing.assert_almost_equal(new_image, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]])
        elif shift_size == 2:
            numpy.testing.assert_almost_equal(new_image, [[[0, 0, 0], [0, 0, 0], [1, 1, 1]]])
        elif shift_size == -1:
            numpy.testing.assert_almost_equal(new_image, [[[2, 2, 2], [3, 3, 3], [0, 0, 0]]])

    def test_random_generators(self):
        def generator(feeding_data):
            return feeding_data.image(), feeding_data.steering_angle + 1

        generator = random_generators(generator, generator, generator)
        image, angle = generator(FeedingData([[[1, 1, 1]]], 2))
        np.testing.assert_almost_equal(image, [[[1, 1, 1]]])
        self.assertEqual(angle, 3)

    def test_pipe_line_generators(self):
        def generator(feeding_data):
            return feeding_data.image(), feeding_data.steering_angle + 1

        generator = pipe_line_generators(generator, generator, generator)
        image, angle = generator(FeedingData([[[1, 1, 1]]], 2))
        np.testing.assert_almost_equal(image, [[[1, 1, 1]]])
        self.assertEqual(angle, 5)

    def test_pipe_line_with_random_generators(self):
        def generator1(feeding_data):
            return feeding_data.image(), feeding_data.steering_angle + 1

        generator = pipe_line_generators(
            random_generators(generator1, generator1, generator1),
            generator1,
            generator1,
            generator1)
        image, angle = generator(FeedingData([[[1, 1, 1]]], 2))
        np.testing.assert_almost_equal(image, [[[1, 1, 1]]])
        self.assertEqual(angle, 6)

    def test_flip_generator(self):
        record = FeedingData(np.array([[[1, 1, 1], [2, 2, 2]]]), 0.1)
        image, angle = flip_generator(record)
        np.testing.assert_almost_equal(image, [[[2, 2, 2], [1, 1, 1]]])
        self.assertEqual(-0.1, angle)

    def test_flip_real_image(self):
        dataset = DriveDataSet("resources/driving_log_mini.csv")
        record = dataset[0]
        center = record.center_image()
        center_flip, center_flip_angle = flip_generator(FeedingData(center, 0.1765823))
        left = record.left_image()
        left_flip, left_flip_angle = flip_generator(FeedingData(left, 0.1765823 + 0.25))
        right = record.right_image()
        right_flip, right_flip_angle = flip_generator(FeedingData(right, 0.1765823 - 0.25))
        Image.fromarray(center).save("resources/flip/center.jpg")
        Image.fromarray(center_flip).save("resources/flip/center_flip.jpg")
        Image.fromarray(left).save("resources/flip/left.jpg")
        Image.fromarray(left_flip).save("resources/flip/left_flip.jpg")
        Image.fromarray(right).save("resources/flip/right.jpg")
        Image.fromarray(right_flip).save("resources/flip/right_flip.jpg")
        self.assertEqual(left_flip_angle, -(0.1765823 + 0.25))
        self.assertEqual(right_flip_angle, -(0.1765823 - 0.25))

    def test_filter(self):
        def generator1(feeding_data):
            return feeding_data.image(), 0.01

        def generator2(feeding_data):
            return feeding_data.image(), 1

        generator = filter_generator(
            random_generators(generator1, generator2)
        )
        image, angle = generator(FeedingData([[[1, 1, 1]]], 2))
        np.testing.assert_almost_equal(image, [[[1, 1, 1]]])
        self.assertEqual(angle, 1, "should always pickup generator2 as it pass the threshold")

