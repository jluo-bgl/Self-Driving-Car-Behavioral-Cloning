import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData, DriveDataSet, FeedingData, AngleTypeWithZeroRecordAllocator
from data_generators import _shift_image, pipe_line_generators, random_generators, flip_generator, filter_generator
from PIL import Image


class TestGeneratorFuncs(unittest.TestCase):
    def test_shift_image(self):
        angle = 1
        image = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        new_image, new_angle, shift_size = _shift_image(image, angle, 3, 0, angle_offset_pre_pixel=0.004)
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


def any_number():
    return 1


def angles(feeding_data_list):
    return [item.steering_angle for item in feeding_data_list]


class TestRecordAllocation(unittest.TestCase):
    def test_record_allocation_angle_type_with_zeros(self):
        data_set = DriveDataSet([
            FeedingData(None, 0.0),
            FeedingData(None, 0.1),
            FeedingData(None, -0.1),
            FeedingData(None, 0.25),
            FeedingData(None, -0.25)
        ])

        allocator = AngleTypeWithZeroRecordAllocator(data_set,
                                                     any_number(), any_number(), any_number(),
                                                     any_number(), any_number(), 0.25)

        self.assertEqual(angles(allocator.zero_angles), [0.0])
        self.assertEqual(angles(allocator.zero_angles_left), [-0.25])
        self.assertEqual(angles(allocator.zero_angles_right), [0.25])
        self.assertEqual(angles(allocator.center_angles), [])
        self.assertEqual(angles(allocator.left_angles), [-0.1])
        self.assertEqual(angles(allocator.right_angles), [0.1])

    def test_record_allocation_angle_type_with_zeros_in_range(self):
        data_set = DriveDataSet([
            FeedingData(None, 0.03),
            FeedingData(None, -0.03),
            FeedingData(None, 0.15),
            FeedingData(None, -0.15),
            FeedingData(None, 0.26),
            FeedingData(None, -0.26)
        ])
        allocator = AngleTypeWithZeroRecordAllocator(data_set,
                                                     any_number(), any_number(), any_number(),
                                                     any_number(), any_number(), 0.25)

        self.assertEqual(angles(allocator.zero_angles), [])
        self.assertEqual(angles(allocator.zero_angles_left), [])
        self.assertEqual(angles(allocator.zero_angles_right), [])
        self.assertEqual(angles(allocator.center_angles), [0.03, -0.03])
        self.assertEqual(angles(allocator.left_angles), [-0.15, -0.26])
        self.assertEqual(angles(allocator.right_angles), [0.15, 0.26])
