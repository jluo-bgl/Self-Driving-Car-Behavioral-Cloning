from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator, \
    RecordRandomAllocator
from data_generators import image_itself, brightness_image_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer
from model import nvidia


def no_modification():
    data_set = DriveDataSet.from_csv(
        "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
        filter_method=drive_record_filter_include_all)

    allocator = RecordRandomAllocator(data_set)
    generator = image_itself
    data_generator = DataGenerator(allocator.allocate, generator)
    model = nvidia(input_shape=data_set.output_shape(), dropout=0.5)
    Trainer(model, learning_rate=0.0001, epoch=10).fit_generator(
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


no_modification()