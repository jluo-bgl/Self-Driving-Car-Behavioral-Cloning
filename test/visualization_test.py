import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData
from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator, \
    drive_record_filter_exclude_small_angles, drive_record_filter_include_all, drive_record_filter_exclude_zeros
from data_generators import image_itself, \
     shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from visualization import Video, Plot
from performance_timer import Timer


class TestVideos(unittest.TestCase):
    def test_gif_for_generator(self):
        dataset = DriveDataSet("../datasets/udacity-sample-track-1/driving_log.csv")

        generator = pipe_line_generators(
            image_itself,
            shift_image_generator
        )
        Video.from_generators("resources/shift_center_images.gif", dataset[60], 0.0, 20, generator)

        # generator = pipe_line_generators(
        #     left_image_generator,
        #     shift_image_generator
        # )
        # Video.from_generators("resources/shift_left_images.gif", dataset[60], 0.2, 20, generator)
        #
        # generator = pipe_line_generators(
        #     right_image_generator,
        #     shift_image_generator
        # )
        # Video.from_generators("resources/shift_right_images.gif", dataset[60], -0.2, 20, generator)

    def test_create_sample_data_video(self):
        Video.from_udacity_sample_data(
            DriveDataSet("../datasets/udacity-sample-track-1/driving_log.csv"),
            "resources/sample_original.mp4")

    def test_create_sample_data_corp_video(self):
        Video.from_udacity_sample_data(
            DriveDataSet("../datasets/udacity-sample-track-1/driving_log.csv", crop_images=True),
            "resources/sample_crop.mp4")


class TestPlot(unittest.TestCase):
    def test_angle_distribution(self):
        with Timer():
            dataset = DriveDataSet(
                "../datasets/udacity-sample-track-1/driving_log.csv", filter_method=drive_record_filter_include_all)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution.jpg")

    def test_angle_distribution_after_filterout_small_angles(self):
        with Timer():
            dataset = DriveDataSet(
                "../datasets/udacity-sample-track-1/driving_log.csv",
                filter_method=drive_record_filter_exclude_small_angles
            )
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution_exclude_small_angles.jpg")

    def test_angle_distribution_after_filterout_zeros(self):
        with Timer():
            dataset = DriveDataSet(
                "../datasets/udacity-sample-track-1/driving_log.csv",
                filter_method=drive_record_filter_exclude_zeros
            )
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution_exclude_zero_angles.jpg")

    def test_angle_distribution_generator(self):
        with Timer():
            dataset = DriveDataSet("../datasets/udacity-sample-track-1/driving_log.csv")
            # dataset = DriveDataSet("resources/driving_log_mini.csv")
        generator = pipe_line_random_generators(
            image_itself,
            shift_image_generator(angle_offset_pre_pixel=0.005),
            flip_generator
        )
        data_generator = DataGenerator(generator)
        image, angles = next(data_generator.next_batch(dataset, 10000))
        plt = Plot.angle_distribution(angles)
        plt.savefig("resources/angle_distribution_generator.jpg")
