from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator
from data_generators import center_image_generator, \
    random_center_left_right_image_generator, \
    left_image_generator, \
    right_image_generator, shift_image_generator, random_generators, pipe_line_generators, \
    flip_generator, filter_generator
from trainer import Trainer

# dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")

# data_generator = DataGenerator(center_left_right_image_generator)
# Trainer(learning_rate=0.0001, epoch=10).fit(data_generator.generate(dataset, batch_size=128))

dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True)
data_generator = DataGenerator(
    random_generators(
        random_center_left_right_image_generator,
        pipe_line_generators(
            random_center_left_right_image_generator,
            shift_image_generator(angle_offset_pre_pixel=0.005)
        ),
        pipe_line_generators(
            random_center_left_right_image_generator,
            shift_image_generator(angle_offset_pre_pixel=0.005),
            flip_generator
        )
    )
)
Trainer(learning_rate=0.002, epoch=80, dropout=0.50, multi_process=True).fit(
    data_generator.generate(dataset, batch_size=128),
    input_shape=dataset.output_shape()
)