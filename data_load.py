import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from performance_timer import Timer


def full_file_name(base_folder, image_file_name):
    return base_folder + "/" + image_file_name.strip()


def read_image_from_file(image_file_name):
    return plt.imread(image_file_name)


def _crop_image(img, new_height=66, new_width=200):
    height, width = img.shape[0], img.shape[1]
    if (new_height >= height) and (new_width >= width):
        return img

    y_start = 60
    x_start = int(width / 2) - int(new_width / 2)

    return img[y_start:y_start + new_height, x_start:x_start + new_width]


class FeedingData(object):
    def __init__(self, image, steering_angle):
        self._image = image
        self.steering_angle = steering_angle

    def image(self):
        return self._image


class DriveRecord(FeedingData):
    """
    One Record is the actual record from CAR, it is a event happened past, immutable and no one is going to
    modify it.
    It has 3 images and steering angle at that time.
    Images will cache to memory after first read (if no one read the file, it won't fill the memory)
    """
    def __init__(self, base_folder, csv_data_frame_row, crop_image=False):
        """

        :param base_folder:
        :param csv_data_frame_row:
        :param crop_image: crop to 66*200 or not, only crop if image larger then 66*200
        """
        # index,center,left,right,steering,throttle,brake,speed
        super().__init__(None, csv_data_frame_row[4])

        self.index = csv_data_frame_row[0]
        self.center_file_name = full_file_name(base_folder, csv_data_frame_row[1])
        self.left_file_name = full_file_name(base_folder, csv_data_frame_row[2])
        self.right_file_name = full_file_name(base_folder, csv_data_frame_row[3])

        self.crop_image = crop_image

        self._center_image = None
        self._left_image = None
        self._right_image = None

    def image(self):
        return self.center_image()

    def center_image(self):
        if self._center_image is None:
            self._center_image = self.read_image(self.center_file_name)

        return self._center_image

    def left_image(self):
        if self._left_image is None:
            self._left_image = self.read_image(self.left_file_name)

        return self._left_image

    def right_image(self):
        if self._right_image is None:
            self._right_image = self.read_image(self.right_file_name)

        return self._right_image

    def read_image(self, file_name):
        image = read_image_from_file(file_name)
        if self.crop_image:
            image = _crop_image(image, 66, 200)
        return image


class DriveDataSet(object):
    """
    DriveDataSet represent multiple Records together, you can access any record by [index] or iterate through
    As it represent past, it's immutable as well
    """
    def __init__(self, file_name, crop_images=False):
        self.base_folder = os.path.split(file_name)[0]
        # center,left,right,steering,throttle,brake,speed
        self.data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        self.records = list(map(
            lambda index: DriveRecord(self.base_folder,
                                      self.data_frame.iloc[[index]].reset_index().values[0],
                                      crop_images),
            range(len(self.data_frame))))

    def __getitem__(self, n):
        return self.records[n]

    def __iter__(self):
        return self.records.__iter__()

    def __len__(self):
        return len(self.records)

    def output_shape(self):
        return self.records[0].image().shape


class DataGenerator(object):
    def __init__(self, custom_generator):
        self.custom_generator = custom_generator

    def generate(self, data_set, batch_size=32):
        input_shape = data_set.output_shape()
        batch_images = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        batch_steering = np.zeros(batch_size)
        while 1:
            for i_batch in range(batch_size):
                index = np.random.randint(len(data_set))
                # with Timer(True):
                x, y = self.custom_generator(data_set[index])
                batch_images[i_batch] = x
                batch_steering[i_batch] = y
            yield batch_images, batch_steering


class DrivingDataLoader(object):
    def __init__(self, file_name, center_img_only):
        self.base_folder = os.path.split(file_name)[0]
        # center,left,right,steering,throttle,brake,speed
        self.data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        # images, angles = self._read_csv(center_img_only)
        # self.images = images
        # self.angles = angles

    def _read_csv(self, center_img_only):
        center_images, center_angles = self._read_image_angles(
            self.data_frame.values[:, 0],
            self.data_frame.values[:, 3]
        )

        all_images, all_angles = center_images, center_angles

        if not center_img_only:
            left_images, left_angles = self._read_image_angles(
                self.data_frame.values[:, 1],
                self.data_frame.values[:, 3] + 0.2
            )
            right_images, right_angles = self._read_image_angles(
                self.data_frame.values[:, 2],
                self.data_frame.values[:, 3] - 0.2
            )

            all_images = np.append(np.append(all_images, left_images, axis=0), right_images, axis=0)
            all_angles = np.append(np.append(all_angles, left_angles, axis=0), right_angles, axis=0)

        return all_images, all_angles

    def _read_image_angles(self, image_files_names, angles):
        images = map(DrivingDataLoader._read_image(self.base_folder), image_files_names)
        return np.array(list(images)), angles

    @staticmethod
    def _read_image(base_folder):
        def read(image_file_name):
            return DrivingDataLoader._read_image_from_file(base_folder + "/" + image_file_name.lstrip())

        return read

    def _read_image_from_file(self, image_file_name):
        return plt.imread(self.base_folder + "/" + image_file_name.strip())

    def images_and_angles(self):
        return self.images, self.angles

    def generate(self, data, batch_size=32):
        batch_images = np.zeros((batch_size, 160, 320, 3))
        batch_steering = np.zeros(batch_size)
        while 1:
            for i_batch in range(batch_size):
                i_line = np.random.randint(len(data))
                line_data = data.iloc[[i_line]].reset_index()
                x, y = self._read_image_from_file(line_data.values[0, 1]), line_data.values[0, 4]
                batch_images[i_batch] = x
                batch_steering[i_batch] = y

            print("generating")
            yield batch_images, batch_steering


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
        *DrivingDataLoader("datasets/udacity-sample-track-1/driving_log.csv", center_img_only=True).images_and_angles())
    assert len(data_provider.images) > 5000
    print("saving data")
    data_provider.save_to_file("datasets/udacity-sample-track-1/driving.p")
    print("done")
