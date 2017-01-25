from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer

data_set = DriveDataSet.from_csv("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
                                 filter_method=drive_record_filter_include_all)
allocator = AngleTypeWithZeroRecordAllocator(
    data_set, left_percentage=20, right_percentage=20,
    zero_percentage=8, zero_left_percentage=6, zero_right_percentage=6,
    left_right_image_offset_angle=0.25)
generator = pipe_line_generators(
    shift_image_generator(angle_offset_pre_pixel=0.002),
    flip_generator,
    brightness_image_generator(0.35),
    shadow_generator
)
data_generator = DataGenerator(allocator, generator)
Trainer(learning_rate=0.0001, epoch=45, dropout=0.5, multi_process=True,
        custom_name="shift_0.002_bright_0.35_angles_35_30_35").fit(
    data_generator.generate(batch_size=256),
    input_shape=data_set.output_shape()
)
