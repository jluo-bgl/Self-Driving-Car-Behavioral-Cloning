import numpy as np


def center_image_generator(data_set_row):
    return data_set_row.center_image(), data_set_row.steering_angle


def center_left_right_image_generator(data_set_row):
    index = np.random.randint(0, 3)
    if index == 0:
        return data_set_row.center_image(), data_set_row.steering_angle
    elif index == 1:
        return data_set_row.left_image(), data_set_row.steering_angle + 0.2
    else:
        return data_set_row.right_image(), data_set_row.steering_angle - 0.2

