from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator, \
    AngleSegmentRecordAllocator, AngleSegment, RecordRandomAllocator
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer
from model import nvidia, nvidia_with_regularizer
import inspect


def is_osx():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        return False
    elif _platform == "darwin":
        return True
    elif _platform == "win32":
        return False


use_multi_process = not is_osx()


def raw_data_centre_image_no_dropout():
    # Create DriveDataSet from csv file, you can specify crop image, using all cameras and which data will included in
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=False, all_cameras_images=False,
        filter_method=drive_record_filter_include_all)
    # What the data distribution will be, below example just randomly return data from data set, so that the
    # distribution will be same with what original data set have
    allocator = RecordRandomAllocator(data_set)
    # what's the data augment pipe line have, this have no pipe line, just the image itself
    augment = image_itself
    # connect allocator and augment together
    data_generator = DataGenerator(allocator.allocate, augment)
    # create the model
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.0)
    # put everthing together, start a real Keras training process with fit_generator
    Trainer(model, learning_rate=0.0001, epoch=10, custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_image_dropout_5():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=False, all_cameras_images=False,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    augment = image_itself
    data_generator = DataGenerator(allocator.allocate, augment)
    # dropout=0.5 was the only difference
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10, custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_left_right_image():
    # all_cameras_images=True was the only difference
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=False, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    generator = image_itself
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10, custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_left_right_image_crop():
    # crop_images=True was the only difference
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    generator = image_itself
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10, custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_left_right_crop_shift():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    # shift_image_generator added in
    generator = shift_image_generator(angle_offset_pre_pixel=0.002)
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    # have to enable multi_process as image generator becomes to bottle neck
    Trainer(model, learning_rate=0.0001, epoch=20, multi_process=use_multi_process,
            custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_left_right_crop_shift_flip():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    # shift_image_generator was the only difference
    generator = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator
    )
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=20, multi_process=use_multi_process,
            custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def raw_data_centre_left_right_crop_shift_flip_brightness_shadow():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    allocator = RecordRandomAllocator(data_set)
    # shift_image_generator was the only difference
    generator = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.35),
        shadow_generator
    )
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=30, multi_process=use_multi_process,
            custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=128)
    )


def segment_left_centre_right():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)

    allocator = AngleTypeWithZeroRecordAllocator(
        data_set, left_percentage=20, right_percentage=20,
        zero_percentage=8, zero_left_percentage=6, zero_right_percentage=6,
        left_right_image_offset_angle=0.25)
    generator = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.25),
        shadow_generator
    )
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10).fit_generator(
        data_generator.generate(batch_size=128)
    )


def segment_normal_distribution_shift_flip_brightness_shadow():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    # fine tune every part of training data so that make it meat std distrubtion
    allocator = AngleSegmentRecordAllocator(
        data_set,
        AngleSegment((-1.5, -0.5), 10),  # big sharp left
        AngleSegment((-0.5, -0.25), 14),  # sharp left
        AngleSegment((-0.25, -0.249), 3),  # sharp turn left (zero right camera)
        AngleSegment((-0.249, -0.1), 10),  # big turn left
        AngleSegment((-0.1, 0), 11),  # straight left
        AngleSegment((0, 0.001), 4),  # straight zero center camera
        AngleSegment((0.001, 0.1), 11),  # straight right
        AngleSegment((0.1, 0.25), 10),  # big turn right
        AngleSegment((0.25, 0.251), 3),  # sharp turn right (zero left camera)
        AngleSegment((0.251, 0.5), 14),  # sharp right
        AngleSegment((0.5, 1.5), 10)  # big sharp right
    )
    # a pipe line with shift -> flip -> brightness -> shadow augment processes
    augment = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.35),
        shadow_generator
    )
    data_generator = DataGenerator(allocator.allocate, augment)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=45, multi_process=use_multi_process,
            custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=256)
    )


def segment_normal_distribution_shift_flip_brightness_shadow_reg():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True, all_cameras_images=True,
        filter_method=drive_record_filter_include_all)
    # fine tune every part of training data so that make it meat std distrubtion
    allocator = AngleSegmentRecordAllocator(
        data_set,
        AngleSegment((-1.5, -0.5), 10),  # big sharp left
        AngleSegment((-0.5, -0.25), 14),  # sharp left
        AngleSegment((-0.25, -0.249), 3),  # sharp turn left (zero right camera)
        AngleSegment((-0.249, -0.1), 10),  # big turn left
        AngleSegment((-0.1, 0), 11),  # straight left
        AngleSegment((0, 0.001), 4),  # straight zero center camera
        AngleSegment((0.001, 0.1), 11),  # straight right
        AngleSegment((0.1, 0.25), 10),  # big turn right
        AngleSegment((0.25, 0.251), 3),  # sharp turn right (zero left camera)
        AngleSegment((0.251, 0.5), 14),  # sharp right
        AngleSegment((0.5, 1.5), 10)  # big sharp right
    )
    # a pipe line with shift -> flip -> brightness -> shadow augment processes
    augment = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.35),
        shadow_generator
    )
    data_generator = DataGenerator(allocator.allocate, augment)
    model = nvidia_with_regularizer(input_shape=data_set.output_shape(), dropout=0.2)
    Trainer(model, learning_rate=0.0001, epoch=45, multi_process=use_multi_process,
            custom_name=inspect.stack()[0][3]).fit_generator(
        data_generator.generate(batch_size=256)
    )


raw_data_centre_image_no_dropout()
raw_data_centre_image_dropout_5()
raw_data_centre_left_right_image()
raw_data_centre_left_right_image_crop()
raw_data_centre_left_right_crop_shift()
raw_data_centre_left_right_crop_shift_flip()
raw_data_centre_left_right_crop_shift_flip_brightness_shadow()
segment_normal_distribution_shift_flip_brightness_shadow()
segment_normal_distribution_shift_flip_brightness_shadow_reg()
