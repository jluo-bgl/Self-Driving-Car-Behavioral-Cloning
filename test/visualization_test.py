import unittest
import numpy.testing
import numpy as np
from data_load import FeedingData, AngleTypeWithZeroRecordAllocator
from data_load import DriveDataSet, DataGenerator, \
    drive_record_filter_exclude_duplicated_small_angles, drive_record_filter_include_all, drive_record_filter_exclude_zeros
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
     shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from visualization import Video, Plot
from performance_timer import Timer
from PIL import Image


class TestVideos(unittest.TestCase):
    def test_gif_for_generator_shift_brightness_shadown(self):
        dataset = DriveDataSet.from_csv("../datasets/udacity-sample-track-1/driving_log.csv")

        generator = pipe_line_generators(
            image_itself,
            shift_image_generator(angle_offset_pre_pixel=0.006),
            brightness_image_generator(0.25),
            shadow_generator
        )
        Video.from_generators("resources/generator_pipe_line.gif", dataset[60], 20, generator)

    def test_gif_for_generator_flip(self):
        dataset = DriveDataSet.from_csv("../datasets/udacity-sample-track-1/driving_log.csv")

        generator = random_generators(
            image_itself,
            flip_generator
        )
        Video.from_generators("resources/generator_flip.gif", dataset[60], 8, generator)

    def test_create_sample_data_video(self):
        Video.from_udacity_sample_data(
            DriveDataSet.from_csv("../datasets/udacity-sample-track-1/driving_log.csv", crop_images=False,
                                  filter_method=drive_record_filter_include_all),
            "resources/sample_original.mp4")

    def test_create_sample_data_corp_video(self):
        Video.from_udacity_sample_data(
            DriveDataSet.from_csv("../datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
                                  filter_method=drive_record_filter_exclude_duplicated_small_angles),
            "resources/sample_crop.mp4")


class TestPlot(unittest.TestCase):
    @staticmethod
    def create_real_dataset(filter_method):
        return DriveDataSet.from_csv(
            "../datasets/udacity-sample-track-1/driving_log.csv",
            filter_method=filter_method,
            fake_image=True
        )

    def test_angle_distribution(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_include_all)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution_original.jpg")

    def test_angle_distribution_after_filterout_small_angles(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_exclude_duplicated_small_angles)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution_exclude_small_angles.jpg")

    def test_angle_distribution_after_filterout_zeros(self):
        dataset = self.create_real_dataset(filter_method=drive_record_filter_exclude_zeros)
        plt = Plot.angle_distribution(dataset.angles())
        plt.savefig("resources/angle_distribution_exclude_zero_angles.jpg")

    def test_angle_distribution_generator_exclude_duplicated_small_angles_30_40_30(self):
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_30_40_30", 100, 256,
            allocator=drive_record_filter_exclude_duplicated_small_angles,
            left_percentage=30,
            right_percentage=30
        )

    def test_angle_distribution_generator_exclude_duplicated_small_angles_40_20_40(self):
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_40_20_40", 100, 256,
            allocator=drive_record_filter_exclude_duplicated_small_angles,
            left_percentage=40,
            right_percentage=40
        )

    def test_angle_distribution_generator_exclude_duplicated_small_angles_40_20_40_004(self):
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_40_20_40_0.004", 100, 256,
            allocator=drive_record_filter_exclude_duplicated_small_angles,
            left_percentage=40,
            right_percentage=40,
            angle_offset_pre_pixel=0.008
        )

    def test_angle_distribution_generator_exclude_duplicated_small_angles_45_10_45(self):
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_45_10_45", 100, 256,
            allocator=drive_record_filter_exclude_duplicated_small_angles,
            left_percentage=45,
            right_percentage=45
        )

    def test_angle_distribution_generator_45_10_45_pipe_line(self):
        data_set = self.create_real_dataset(filter_method=drive_record_filter_exclude_duplicated_small_angles)
        allocator = AngleTypeWithZeroRecordAllocator(data_set, 20, 20, 15, 15, 15, 0.25)
        generator = pipe_line_generators(
            shift_image_generator(angle_offset_pre_pixel=0.002),
            flip_generator,
            brightness_image_generator(0.25)
        )
        self._angle_distribution(
            "angle_distribution_generator_exclude_duplicated_small_angles_40_20_40_pipe_line", 100, 256,
            allocator=allocator.allocate,
            left_percentage=40,
            right_percentage=40,
            angle_offset_pre_pixel=0.006,
            generator=generator
        )

    def _angle_distribution(
            self, name, batches, batch_size, allocator,
            left_percentage=30, right_percentage=30, angle_offset_pre_pixel=0.002, generator=None
    ):

        if generator is None:
            generator = pipe_line_random_generators(
                image_itself,
                shift_image_generator(angle_offset_pre_pixel=angle_offset_pre_pixel),
                flip_generator
            )
        data_generator = DataGenerator(allocator, generator)
        angles = np.array([])
        for index in range(batches):
            print("batch {} / {}".format(index, batches))
            _, _angles = next(data_generator.generate(batch_size=batch_size))
            angles = np.append(angles, _angles)

        plt = Plot.angle_distribution(angles)
        plt.savefig("resources/{}.jpg".format(name))
