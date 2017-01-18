import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData
from data_generators import _shift_image, pipe_line_generators, random_generators


class TestGeneratorFuncs(unittest.TestCase):
    def test_shift_image(self):
        angle = 1
        image = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        new_image, new_angle, shift_size = _shift_image(image, angle, 3, 0)
        self.assertEqual(new_angle, angle + shift_size * 0.002)
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
        def generator(feeding_data):
            return feeding_data.image(), feeding_data.steering_angle + 1

        generator = pipe_line_generators(
            random_generators(generator, generator, generator),
            generator,
            generator,
            generator)
        image, angle = generator(FeedingData([[[1, 1, 1]]], 2))
        np.testing.assert_almost_equal(image, [[[1, 1, 1]]])
        self.assertEqual(angle, 6)

