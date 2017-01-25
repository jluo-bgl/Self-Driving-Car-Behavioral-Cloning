import unittest
import numpy.testing
import os
import numpy as np
from data_load import DriveDataSet

class TestDriveDataSet(unittest.TestCase):
    def test_data_generator_should_able_to_extend_easily(self):
        dataset = DriveDataSet.from_csv("resources/driving_log_mini.csv")
        self.assertEqual(len(dataset), 6)

