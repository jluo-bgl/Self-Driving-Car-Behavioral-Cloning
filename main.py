import load_data
from model import NVIDA, nvida1
from data_load import DriveDataProvider
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from trainer import Trainer

data_provider = DriveDataProvider.load_from_file("datasets/udacity-sample-track-1/driving_data.p")
Trainer(data_provider).fit()