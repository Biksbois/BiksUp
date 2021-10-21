import unittest2
from save_epoch import save_average
from parameterized import parameterized
import os

class TestKeyParser(unittest2.TestCase):
    @parameterized.expand([
        ['21_10_2021-12_08_15', ['0101']]
    ])
    def test_avg_csv(self, folder, keys):
        root_path = os.path.join('test', 'test_csv')
        avg_path = os.path.join(root_path, 'AVG')
        save_average(3, folder, keys, root_path=root_path, save_root=avg_path)
        self.assertTrue(True)