import unittest2
from metadata import get_metadata_list, metadataObj, is_valid_key, get_key_mutations, get_key_count, DEFAULT_BOOL, contains_illigal_key
from parameterized import parameterized


class TestKeyParser(unittest2.TestCase):
    @parameterized.expand([
        [["00_1", "00_"], 1],
        [["00_1", "00_", "__"], 2],
        [["___"], 3],
        [["000", "000"], 1],
        [["000", "00_"], 1],
        [["0", "01"], 0]
    ])
    def test_get_key_length(self, actual, expected_length):
        actual_list = get_metadata_list(actual, False, False, do_log=False)
        
        self.assertEqual(len(actual_list), expected_length)
    
    @parameterized.expand([
        ["000"],
        ["100"],
        ["010"],
        ["011"],
    ])
    def test_object_generation_three(self, key):
        actual_obj = metadataObj(key)
        
        boolList = [True if x == '1' else False for x in key]
        
        expected_obj = metadataObj()
        
        expected_obj.meta_dict['nonhybrid'] = boolList[0]
        expected_obj.meta_dict['attention'] = boolList[1]
        expected_obj.meta_dict['local'] = boolList[2]
        
        self.assertEqual(actual_obj, expected_obj)
    
    @parameterized.expand([
        ['001'],
        ['__'],
        ['1'],
        [''],
        ['111']
    ])
    def test_valid_key(self, input_key):
        actual = is_valid_key(input_key, 10, do_log=False)
        
        self.assertTrue(actual)
    
    @parameterized.expand([
        ['001a'],
        ['____-'],
        ['11111111111'],
        [' '],
    ])
    def test_not_valid_key(self, input):
        actual = is_valid_key(input, 10, do_log=False)
        self.assertFalse(actual)

    @parameterized.expand([
        ['00000'],
        ['0000'],
        ['010'],
        ['000']
    ])
    def test_contains_not_illigal_combination(self, input_key):
        actual = contains_illigal_key(input_key, do_log=False)
        self.assertFalse(actual)
    
    @parameterized.expand([
        ['100'],
        ['011'],
        ['0111'],
        ['110'],
        ['001'],
        ['001000'],
        ['0111'],
    ])
    def test_contains_illigal_combination(self, input_key):
        actual = contains_illigal_key(input_key, do_log=False)
        self.assertTrue(actual)
    
    @parameterized.expand([
        ['1111', ['1111']],
        ['0', ['0']],
        ['_', ['0', '1']],
        ['_111', ['1111', '0111']],
        ['00__', ['0000', '0011', '0010', '0001']],
        ['', [  '1']],
        ['___', ['000', '111', '001', '010', '100', '011', '110', '101']]
    ])
    def test_get_mutations(self, input, expected):
        key_count = get_key_count()
        char_to_pad = '1' if DEFAULT_BOOL == True else '0' 
        
        expected = [x.ljust(key_count, char_to_pad) for x in expected]
        
        actual = get_key_mutations(input)
        actual = [x.ljust(key_count, char_to_pad) for x in actual]

        actual = sorted(actual)
        expected = sorted(expected)
        
        self.assertListEqual(actual, expected)


if __name__ == '__main__':
    unittest2.main()