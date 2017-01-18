import unittest
import numpy.testing
import os
import numpy as np
from data_load import DrivingDataLoader
from data_load import DriveDataProvider, DataGenerator, Record, DriveDataSet


class TestDataLoad(unittest.TestCase):
    def test_load_data_center_only(self):
        loader = DrivingDataLoader("resources/driving_log_mini.csv", center_img_only=True)
        images, angles = loader.images_and_angles()
        assert len(images) == 2
        assert len(angles) == 2
        numpy.testing.assert_almost_equal([0.1765823, 0.1765823], angles)
        self.assertTupleEqual((2, 160, 320, 3), images.shape)

    def test_load_data(self):
        loader = DrivingDataLoader("resources/driving_log_mini.csv", center_img_only=False)
        images, angles = loader.images_and_angles()
        assert len(images) == 6
        assert len(angles) == 6
        numpy.testing.assert_almost_equal([0.1765823, 0.1765823, 0.3765823, 0.37658230, -0.0234177, -0.0234177], angles)
        self.assertTupleEqual((6, 160, 320, 3), images.shape)


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

    def test_integration_load_save_file(self):
        loader = DrivingDataLoader("resources/driving_log_mini.csv")
        provider = DriveDataProvider(*loader.images_and_angles())
        filename = "resources/data_provider_test_integration.p"
        provider.save_to_file(filename)

        provider2 = DriveDataProvider.load_from_file(filename)
        self.assertEqual(len(provider2.images), 6)
        np.testing.assert_almost_equal(provider.images, provider2.images)


class TestDriveDataSet(unittest.TestCase):
    def test_data_generator_should_able_to_extend_easily(self):
        def center_image_generator(dataset_row):
            return dataset_row.center_image(), dataset_row.steering_angle

        dataset = DriveDataSet("resources/driving_log_mini.csv")
        data_generator = DataGenerator(center_image_generator)
        generator = data_generator.generate(dataset, 4)
        x, y = next(generator)
        self.assertEqual(len(x), 4, "should have 4 images")
        np.testing.assert_almost_equal(y, [0.1765823, 0.1765823, 0.1765823, 0.1765823])

