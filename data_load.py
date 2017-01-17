import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
# % time
# white_clip.write_videofile(white_output, audio=False)


class DrivingDataLoader(object):
    def __init__(self, file_name):
        center_images, angles = self._read_csv(file_name)
        self.center_images = center_images
        self.angles = angles

    @staticmethod
    def _read_csv(file_name):
        data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        center_images_files = data_frame.values[:, 0]
        angles = data_frame.values[:, 3]
        center_images = map(DrivingDataLoader._read_image(os.path.split(file_name)[0]), center_images_files)
        return np.array(list(center_images)), angles

    @staticmethod
    def _read_image(base_folder):
        def read(image_file_name):
            return plt.imread(base_folder + "/" + image_file_name)

        return read

    def images_and_angles(self):
        return self.center_images, self.angles


class DriveDataProvider(object):
    """
    provide data to neural network
    """

    def __init__(self, images, angles):
        self.images = images
        self.angles = angles

    def save_to_file(self, file_name):
        data = {
            "images": self.images,
            "angles": self.angles
        }
        with open(file_name, mode='wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, mode='rb') as f:
            data = pickle.load(f)

        return DriveDataProvider(
            images=data["images"],
            angles=data["angles"]
        )

    @classmethod
    def from_other_provider(cls, data_provider):
        return cls(data_provider.images, data_provider.angles)


if __name__ == '__main__':
    print("loading data")
    data_provider = DriveDataProvider(
        *DrivingDataLoader("datasets/udacity-sample-track-1/driving_log.csv").images_and_angles())
    assert len(data_provider.images) > 5000
    print("saving data")
    data_provider.save_to_file("datasets/udacity-sample-track-1/driving_data.p")
    print("done")
