import unittest2
from save_epoch import save_std
from save_epoch import save_average
from metadata import get_metadata_list
from parameterized import parameterized
import os
from pandas.util.testing import assert_frame_equal
import pandas as pd
from icecream import ic

class TestKeyParser(unittest2.TestCase):
    @parameterized.expand([
        ['21_10_2021-12_08_15', ['01000']]
    ])
    def test_avg_csv(self, folder, keys):
        root_path = os.path.join('test', 'test_csv')
        avg_path = os.path.join(root_path, 'AVG')
        std_path = os.path.join(root_path, 'STD')
        
        metadata = get_metadata_list(keys, False, False, do_log=True)

        save_average(2, folder, metadata, root_path=root_path, save_root=avg_path, second_root=std_path)
        
        expected_avg_path = os.path.join(avg_path, folder, 'k_' +  keys[0] + '_i_2' + '.csv')
        actual_avg_path = os.path.join(avg_path, folder, 'k_' + keys[0] + '_i_2_EXPECTED' + '.csv')
        
        expected_avg_pd = pd.read_csv(expected_avg_path)
        actual_avg_pd = pd.read_csv(actual_avg_path)
        
        actual_std_path = os.path.join(std_path, folder, 'k_' +keys[0] + '_i_2' + '.csv')
        expected_std_path = os.path.join(std_path, folder, 'k_' +keys[0] + '_i_2_EXPECTED' + '.csv')
        
        expected_avg_path = "test/test_csv/STD/21_10_2021-12_08_15/k_01000_i_2_EXPECTED.csv"
        actual_std_path = "test/test_csv/STD/21_10_2021-12_08_15/k_01000_i_2.csv"
        expected_std_pd = pd.read_csv(expected_std_path)
        actual_std_pd = pd.read_csv(actual_std_path)
        
        assert_frame_equal(expected_avg_pd, actual_avg_pd)
        assert_frame_equal(expected_std_pd, actual_std_pd)