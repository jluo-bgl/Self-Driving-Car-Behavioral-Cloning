from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator, record_allocation_angle_type
from data_generators import image_itself, \
    shift_image_generator, random_generators, pipe_line_generators, pipe_line_random_generators, flip_generator
from trainer import Trainer

# dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")

# data_generator = DataGenerator(center_left_right_image_generator)
# Trainer(learning_rate=0.0001, epoch=10).fit(data_generator.generate(dataset, batch_size=128))

dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True)
data_generator = DataGenerator(
    pipe_line_random_generators(
        image_itself,
        shift_image_generator(angle_offset_pre_pixel=0.005),
        flip_generator
    )
)
Trainer(learning_rate=0.0001, epoch=10, dropout=0.5).fit(
    data_generator.generate(dataset, batch_size=128, record_allocation_method=record_allocation_angle_type(40, 40)),
    input_shape=dataset.output_shape()
)
