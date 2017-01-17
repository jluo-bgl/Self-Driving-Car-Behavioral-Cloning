import unittest
import numpy.testing
import os
import numpy as np
from data_load import DrivingDataLoader
from data_load import DriveDataProvider


class TestDataLoad(unittest.TestCase):
    def test_load_data(self):
        loader = DrivingDataLoader("resources/driving_log_mini.csv")
        images, angles = loader.images_and_angles()
        assert len(images) == 2
        assert len(angles) == 2
        numpy.testing.assert_almost_equal([0.1765823, 0.1765823], angles)
        self.assertTupleEqual((2, 160, 320, 3), images.shape)


class TestDriveDataProvider(unittest.TestCase):
    def test_save_load_file(self):
        filename = "resources/data_provider_test.p"
        images = [[1, 2]]
        angles = [0.1]
        provider = DriveDataProvider(images, angles)
        provider.save_to_file(filename)

        provider2 = DriveDataProvider.load_from_file(filename)
        numpy.testing.assert_almost_equal(images, provider2.images)
        numpy.testing.assert_almost_equal(angles, provider2.angles)
