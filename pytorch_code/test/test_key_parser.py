import unittest2
from metadata import get_metadata_list, metadataObj, is_valid_key, get_key_mutations, get_key_count, DEFAULT_BOOL
from parameterized import parameterized


class TestKeyParser(unittest2.TestCase):
    @parameterized.expand([
        [["0011_11", "000_"], 4],
        [["0011_11", "000_", "__"], 7],
        [["_____"], 32],
        [["000", "000"], 1],
        [["000", "00_"], 2],
        [["0", "01"], 1]
    ])
    def test_get_key_length(self, actual, expected_length):
        actual_list = get_metadata_list(actual)
        
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
        
        expected_obj.meta_dict['key_01'] = boolList[0]
        expected_obj.meta_dict['key_02'] = boolList[1]
        expected_obj.meta_dict['key_03'] = boolList[2]
        
        self.assertEqual(actual_obj, expected_obj)
    
    @parameterized.expand([
        ['0010'],
        ['____'],
        ['1'],
        [''],
        ['1111111111']
    ])
    def test_valid_key(self, input_key):
        actual = is_valid_key(input_key, 10)
        
        self.assertTrue(actual)
    
    @parameterized.expand([
        ['001a'],
        ['____-'],
        ['11111111111'],
        [' '],
    ])
    def test_not_valid_key(self, input):
        actual = is_valid_key(input, 10)
        self.assertFalse(actual)

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