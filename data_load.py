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


def drive_record_filter_exclude_duplicated_small_angles(last_added_records, current_drive_record):
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
    def __init__(self, records):
        self.records = records

    @classmethod
    def from_csv(cls, file_name, crop_images=False, fake_image=False,
                 filter_method=drive_record_filter_exclude_duplicated_small_angles):
        records, _, _, _ = cls.read_from_csv(file_name, crop_images, fake_image, filter_method)
        return cls(records)

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
    def read_from_csv(file_name, crop_images=False, fake_image=False,
                      filter_method=drive_record_filter_exclude_duplicated_small_angles):
        base_folder = os.path.split(file_name)[0]
        # center,left,right,steering,throttle,brake,speed
        data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        drive_records = list(map(
            lambda index: DriveRecord(base_folder,
                                      data_frame.iloc[[index]].reset_index().values[0],
                                      crop_images,
                                      fake_image=fake_image),
            range(len(data_frame))))
        records = DriveDataSet.drive_record_to_feeding_data(drive_records, filter_method)

        return records, drive_records, data_frame, base_folder


def _random_access_list(data_list, size):
    random_ids = np.random.randint(0, len(data_list), size)
    return [data_list[index] for index in random_ids]


class RecordAllocator(object):
    def __init__(self, data_set):
        self.data_set = data_set

    def allocate(self, epoch, batch_number, batch_size):
        raise NotImplementedError("Please Implement this method")


class RecordRandomAllocator(RecordAllocator):
    def allocate(self, epoch, batch_number, batch_size):
        return _random_access_list(self.data_set.records, batch_size)


class RecordAngleTypeAllocator(RecordAllocator):
    def __init__(self, data_set, left_percentage, right_percentage):
        self.left_percentage = left_percentage
        self.right_percentage = right_percentage
        self.data_set = data_set
        self.center_angles = data_set.straight_records
        self.left_angles = data_set.left_records
        self.right_angles = data_set.right_records

    def allocate(self, epoch, batch_number, batch_size):
        left_size = batch_size * self.left_percentage // 100
        right_size = batch_size * self.right_percentage // 100
        center_size = batch_size - left_size - right_size

        return _random_access_list(self.center_angles, center_size) + \
               _random_access_list(self.left_angles, left_size) + \
               _random_access_list(self.right_angles, right_size)


class AngleTypeWithZeroRecordAllocator(RecordAllocator):
    def __init__(self, data_set,
                 left_percentage, right_percentage,
                 zero_percentage, zero_left_percentage, zero_right_percentage,
                 left_right_image_offset_angle):
        self.left_percentage = left_percentage
        self.right_percentage = right_percentage
        self.zero_percentage = zero_percentage
        self.zero_left_percentage = zero_left_percentage
        self.zero_right_percentage = zero_right_percentage
        self.data_set = data_set

        feeding_data_list = data_set.records
        float_margin = 0.001
        straight_angle = 0.1
        self.zero_angles = self._records_of_range(feeding_data_list, 0, float_margin)
        self.zero_angles_left = self._records_of_range(feeding_data_list, -left_right_image_offset_angle, float_margin)
        self.zero_angles_right = self._records_of_range(feeding_data_list, left_right_image_offset_angle, float_margin)
        self.center_angles = self._straight_records(feeding_data_list, straight_angle, float_margin)
        self.left_angles = self._left_records(
            feeding_data_list, straight_angle, float_margin, left_right_image_offset_angle)
        self.right_angles = self._right_records(
            feeding_data_list, straight_angle, float_margin, left_right_image_offset_angle)
        total_records = len(self.zero_angles) + \
                        len(self.zero_angles_left) + len(self.zero_angles_right) + \
                        len(self.left_angles) + len(self.right_angles) + len(self.center_angles)
        assert len(feeding_data_list) == total_records

    @staticmethod
    def _records_of_range(records, center, offset):
        return [record for record in records if
                center - offset < record.steering_angle < center + offset]

    @staticmethod
    def _straight_records(records, straight_angle, zero_angle):
        return [record for record in records if
                zero_angle <= abs(record.steering_angle) < straight_angle]

    @staticmethod
    def _left_records(records, straight_angle, zero_angle, left_right_image_offset_angle):
        return [record for record in records if
                (record.steering_angle <= -left_right_image_offset_angle - zero_angle) or
                (-left_right_image_offset_angle + zero_angle <= record.steering_angle <= -straight_angle)
                ]

    @staticmethod
    def _right_records(records, straight_angle, zero_angle, left_right_image_offset_angle):
        return [record for record in records if
                (record.steering_angle >= left_right_image_offset_angle + zero_angle) or
                (left_right_image_offset_angle - zero_angle >= record.steering_angle >= straight_angle)
                ]

    def allocate(self, epoch, batch_number, batch_size):
        left_size = batch_size * self.left_percentage // 100
        right_size = batch_size * self.right_percentage // 100
        zero_size = batch_size * self.zero_percentage // 100
        zero_left_size = batch_size * self.zero_left_percentage // 100
        zero_right_size = batch_size * self.zero_right_percentage // 100
        center_size = batch_size - left_size - right_size - zero_size - zero_left_size - zero_right_size

        return _random_access_list(self.center_angles, center_size) + \
               _random_access_list(self.left_angles, left_size) + \
               _random_access_list(self.right_angles, right_size) + \
               _random_access_list(self.zero_angles, zero_size) + \
               _random_access_list(self.zero_angles_left, zero_left_size) + \
               _random_access_list(self.zero_angles_right, zero_right_size)


class DataGenerator(object):
    def __init__(self, record_allocation_method, custom_generator):
        assert record_allocation_method is not None
        assert custom_generator is not None
        self.record_allocation_method = record_allocation_method
        self.custom_generator = custom_generator

    def generate(self, batch_size=32):
        epoch = -1
        batch = -1
        while True:
            epoch += 1
            batch += 1
            selected_records = self.record_allocation_method(epoch, batch, batch_size)
            input_shape = selected_records[0].image().shape
            batch_images = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
            batch_steering = np.zeros(batch_size)
            i_record = 0
            for record in selected_records:
                for retry in range(50):
                    x, y = self.custom_generator(record)
                    batch_images[i_record] = x
                    batch_steering[i_record] = y
                    if abs(y) < 1.:
                        break
                    if retry > 20:
                        print("angle {} retrying {}".format(y, retry))
                i_record += 1
            yield batch_images, batch_steering
