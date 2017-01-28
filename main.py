from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator, \
    AngleSegmentRecordAllocator, AngleSegment, RecordRandomAllocator
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer
from model import nvidia


def is_osx():
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        return False
    elif _platform == "darwin":
        return True
    elif _platform == "win32":
        return False


use_multi_process = not is_osx()


def raw_data_centre_image_only():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=False, all_cameras_images=False,
        filter_method=drive_record_filter_include_all)

    allocator = RecordRandomAllocator(data_set)
    generator = image_itself
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10, custom_name=raw_data_centre_image_only.__name__).fit_generator(
        data_generator.generate(batch_size=128)
    )


def segment_left_centre_right():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
        filter_method=drive_record_filter_include_all)

    allocator = AngleTypeWithZeroRecordAllocator(
        data_set, left_percentage=20, right_percentage=20,
        zero_percentage=8, zero_left_percentage=6, zero_right_percentage=6,
        left_right_image_offset_angle=0.25)
    generator = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.25)
    )
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10).fit_generator(
        data_generator.generate(batch_size=128)
    )


def segment():
    data_set = DriveDataSet.from_csv("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
                                     filter_method=drive_record_filter_include_all)
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
    generator = pipe_line_generators(
        shift_image_generator(angle_offset_pre_pixel=0.002),
        flip_generator,
        brightness_image_generator(0.35),
        shadow_generator
    )
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=45, multi_process=use_multi_process,
            custom_name="bigger_angle_shift_0.002_bright_0.35_angles_35_30_35").fit_generator(
        data_generator.generate(batch_size=256)
    )


raw_data_centre_image_only()
