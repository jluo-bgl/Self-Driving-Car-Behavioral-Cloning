import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData, DriveDataSet, FeedingData, AngleTypeWithZeroRecordAllocator, \
    AngleSegmentRecordAllocator, AngleSegment
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


def same_range_angles(allocator, allocated_records, angle_to_exam):
    same_range_records = allocator.allocated_records_count(allocated_records, angle_to_exam)[1]
    return [item.steering_angle for item in same_range_records]


class TestRecordAllocation(unittest.TestCase):
    def test_AngleSegment_in_range_should_return_true_if_angle_in_segment(self):
        self.assertTrue(AngleSegment((-0.1, 1), any_number()).in_range(-0.1), "lower eager inclusive")
        self.assertTrue(AngleSegment((-0.1, 1), any_number()).in_range(-0.05), "in range")
        self.assertFalse(AngleSegment((-0.1, 1), any_number()).in_range(1), "up eager exclusive")

    def test_record_allocation_angle_type_with_zeros(self):
        data_set = DriveDataSet([
            FeedingData(None, 0.0),
            FeedingData(None, 0.1),
            FeedingData(None, -0.1),
            FeedingData(None, 0.25),
            FeedingData(None, -0.25),
            FeedingData(None, -1),
            FeedingData(None, 0.001),
            FeedingData(None, 0.251),
            FeedingData(None, -0.249),
        ])

        allocator = AngleSegmentRecordAllocator(
            data_set,
            AngleSegment((-1.0, -0.25), 10),
            AngleSegment((-0.25, -0.249), 10),
            AngleSegment((-0.249, -0.1), 10),
            AngleSegment((-0.1, 0), 10),
            AngleSegment((0, 0.001), 10),
            AngleSegment((0.001, 0.1), 10),
            AngleSegment((0.1, 0.25), 10),
            AngleSegment((0.25, 0.251), 10),
            AngleSegment((0.251, 1.001), 10),
            AngleSegment((1.001, 2.0), 10))
        records = allocator.allocate(any_number(), any_number(), len(data_set))
        self.assertEqual(same_range_angles(allocator, records, 0), [0.0])
        self.assertEqual(same_range_angles(allocator, records, 0.1), [0.1])
        self.assertEqual(same_range_angles(allocator, records, -0.1), [-0.1])
        self.assertEqual(same_range_angles(allocator, records, 0.25), [0.25])
        self.assertEqual(same_range_angles(allocator, records, -0.25), [-0.25])
        self.assertEqual(same_range_angles(allocator, records, 1), [0.251])
        self.assertEqual(same_range_angles(allocator, records, -1), [-1])
        self.assertEqual(same_range_angles(allocator, records, 0.001), [0.001])
        self.assertEqual(same_range_angles(allocator, records, -0.001), [-0.1])
        self.assertEqual(same_range_angles(allocator, records, 0.252), [0.251])
        self.assertEqual(same_range_angles(allocator, records, -0.249), [-0.249])

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
