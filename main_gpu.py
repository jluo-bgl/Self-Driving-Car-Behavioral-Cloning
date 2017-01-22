from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator
from data_generators import image_itself, \
    shift_image_generator, random_generators, pipe_line_generators, \
    flip_generator, filter_generator, pipe_line_random_generators
from trainer import Trainer

dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True)
data_generator = DataGenerator(
    pipe_line_random_generators(
        image_itself,
        shift_image_generator(angle_offset_pre_pixel=0.005),
        flip_generator
    )
)
Trainer(learning_rate=0.0001, epoch=20, dropout=0.50, multi_process=True).fit(
    data_generator.generate(dataset, batch_size=128),
    input_shape=dataset.output_shape()
)