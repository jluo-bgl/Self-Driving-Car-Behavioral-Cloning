import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from performance_timer import Timer

CROP_HEIGHT = 66
CROP_WIDTH = 200


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


def _flatten(listoflists):
    return [item for list in listoflists for item in list]


class FeedingData(object):
    def __init__(self, image, steering_angle):
        self._image = image
        self.steering_angle = steering_angle

    def image(self):
        return self._image


class DriveRecord(object):
    """
    One Record is the actual record from CAR, it is a event happened past, immutable and no one is going to
    modify it.
    It has 3 images and steering angle at that time.
    Images will cache to memory after first read (if no one read the file, it won't fill the memory)
    """
    def __init__(self, base_folder, csv_data_frame_row, crop_image=False, fake_image=False):
        """

        :param base_folder:
        :param csv_data_frame_row:
        :param crop_image: crop to 66*200 or not, only crop if image larger then 66*200
        """
        # index,center,left,right,steering,throttle,brake,speed
        self.index = csv_data_frame_row[0]
        self.center_file_name = full_file_name(base_folder, csv_data_frame_row[1])
        self.left_file_name = full_file_name(base_folder, csv_data_frame_row[2])
        self.right_file_name = full_file_name(base_folder, csv_data_frame_row[3])
        self.steering_angle = csv_data_frame_row[4]

        self.crop_image = crop_image
        self.fake_image = fake_image

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
        if self.fake_image:
            return np.array([[[1, 1, 1]]]).astype(np.uint8)
        image = read_image_from_file(file_name)
        if self.crop_image:
            image = _crop_image(image, 66, 200)
        return image


def drive_record_filter_include_all(last_added_records, current_drive_record):
    return current_drive_record


def drive_record_filter_exclude_zeros(last_added_records, current_drive_record):
    if abs(current_drive_record.steering_angle) > 0.02:
        return current_drive_record
    else:
        return None


def drive_record_filter_exclude_small_angles(last_added_records, current_drive_record):
    """
    The filter method which drive record you want add into training samples
    :param last_added_records: last x records we just added in, this could change, you have to check the length
    :param current_drive_record: the DriveRecord do you want add in
    :return: DriveRecord to add into training sample, None if don't want, you can change the DriveRecord if you want
    """
    if abs(current_drive_record.steering_angle) < 0.01:
        how_many_small_angles = 0
        for record in last_added_records:
            if abs(record.steering_angle) < 0.01:
                how_many_small_angles += 1
        if how_many_small_angles >= 1:
            return None
    return current_drive_record


class DriveDataSet(object):
    """
    DriveDataSet represent multiple Records together, you can access any record by [index] or iterate through
    As it represent past, it's immutable as well
    """
    def __init__(self, file_name, crop_images=False, fake_image=False,
                 filter_method=drive_record_filter_exclude_small_angles):
        self.base_folder = os.path.split(file_name)[0]
        # center,left,right,steering,throttle,brake,speed
        self.data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        self.drive_records = list(map(
            lambda index: DriveRecord(self.base_folder,
                                      self.data_frame.iloc[[index]].reset_index().values[0],
                                      crop_images,
                                      fake_image=fake_image),
            range(len(self.data_frame))))
        self.records = self.drive_record_to_feeding_data(self.drive_records, filter_method)
        straight, left, right = self.records_to_straight_left_right(self.records)
        self.straight_records = straight
        self.left_records = left
        self.right_records = right

    def __getitem__(self, n):
        return self.records[n]

    def __iter__(self):
        return self.records.__iter__()

    def __len__(self):
        return len(self.records)

    def angles(self):
        return [feeding_data.steering_angle for feeding_data in self.records]

    def output_shape(self):
        return self.records[0].image().shape

    @staticmethod
    def drive_record_to_feeding_data(records, filter_method):
        # def process_stack(image):
        #     distorted_image = image
        #     if crop_images:
        #         distorted_image = tf.image.resize_image_with_crop_or_pad(image, CROP_HEIGHT, CROP_WIDTH)
        #
        #     return distorted_image

        feeding_data_list = []
        last_5_added = []
        for driving_record in records:
            filtered_record = filter_method(last_5_added, driving_record)
            if filtered_record is not None:
                if len(last_5_added) >= 5:
                    last_5_added.pop(0)
                last_5_added.append(driving_record)

                if abs(driving_record.steering_angle) <= 1.0:
                    feeding_data_list.append(FeedingData(driving_record.center_image(), driving_record.steering_angle))
                if abs(driving_record.steering_angle + 0.25) <= 1.0:
                    feeding_data_list.append(FeedingData(driving_record.left_image(), driving_record.steering_angle + 0.25))
                if abs(driving_record.steering_angle - 0.25) <= 1.0:
                    feeding_data_list.append(FeedingData(driving_record.right_image(), driving_record.steering_angle - 0.25))

        # tensor = tf.map_fn(lambda image: process_stack(image), records, dtype=dtypes.uint8)
        # return tf.Session().run(tensor)
        return feeding_data_list

    @staticmethod
    def records_to_straight_left_right(feeding_data_list):
        straight_angle = 0.1
        straight = [record for record in feeding_data_list if -straight_angle <= record.steering_angle <= straight_angle]
        left = [record for record in feeding_data_list if record.steering_angle > straight_angle]
        right = [record for record in feeding_data_list if record.steering_angle < -straight_angle]
        return straight, left, right


def _random_access_list(data_list, size):
    random_ids = np.random.randint(0, len(data_list), size)
    return [data_list[index] for index in random_ids]


def record_allocation_random(batch_size, all_records, left_angles, center_angles, right_angles):
    return _random_access_list(all_records, batch_size)


def record_allocation_angle_type(left_percentage, right_percentage):
    def _impl(batch_size, all_records, left_angles, center_angles, right_angles):
        left_size = batch_size * left_percentage // 100
        right_size = batch_size * right_percentage // 100
        center_size = batch_size - left_size - right_size

        return _random_access_list(center_angles, center_size) + \
               _random_access_list(left_angles, left_size) + \
               _random_access_list(right_angles, right_size)
    return _impl


class DataGenerator(object):
    def __init__(self, custom_generator):
        self.custom_generator = custom_generator

    def generate(self, data_set, batch_size=32, record_allocation_method=record_allocation_random):
        input_shape = data_set.output_shape()
        batch_images = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        batch_steering = np.zeros(batch_size)
        while True:
            selected_records = record_allocation_method(
                batch_size, data_set.records,
                data_set.left_records, data_set.straight_records, data_set.right_records)
            i_batch = 0
            for record in selected_records:
                for retry in range(50):
                    x, y = self.custom_generator(record)
                    batch_images[i_batch] = x
                    batch_steering[i_batch] = y
                    if abs(y) < 1.:
                        break
                    if retry > 20:
                        print("angle {} retrying {}".format(y, retry))
                i_batch += 1
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
