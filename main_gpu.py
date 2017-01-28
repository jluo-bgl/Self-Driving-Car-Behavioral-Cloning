from data_load import DriveDataSet, DataGenerator, drive_record_filter_include_all, AngleTypeWithZeroRecordAllocator, \
    AngleSegmentRecordAllocator, AngleSegment
from data_generators import image_itself, brightness_image_generator, shadow_generator, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer
from model import nvidia

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
Trainer(learning_rate=0.0001, epoch=45, dropout=0.5, multi_process=True,
        custom_name="bigger_angle_shift_0.002_bright_0.35_angles_35_30_35").fit(
    data_generator.generate(batch_size=256),
    input_shape=data_set.output_shape()
)
