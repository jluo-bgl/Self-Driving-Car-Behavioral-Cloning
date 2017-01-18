import load_data
from model import NVIDA, nvida1
from data_load import DriveDataProvider, DrivingDataLoader, DriveDataSet, DataGenerator
from data_generators import center_image_generator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from trainer import Trainer

dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")
data_generator = DataGenerator(center_image_generator)
Trainer(None).fit(data_generator.generate(dataset, batch_size=128))
