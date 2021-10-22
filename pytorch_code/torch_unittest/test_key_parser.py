# import unittest2
# from metadata import get_metadata_list, metadataObj, is_valid_key
# from parameterized import parameterized


# class TestKeyParser(unittest2.TestCase):
#     # @parameterized.expand([
#     #     [["0011-11", "000_"], ['0011011', '0011111', '0000', '0001']],
#     # ])
#     # def test_get_keys(self, input, expected_list):
#     #     actual_list = []
#     #     expected_list = [metadataObj(x) for x in expected_list]

#     #     for key in input:
#     #         actual_list.append(metadataObj(key))
        
#     #     self.assertEqual(len(actual_list), len(expected_list))
#     #     self.assertListEqual(expected_list, actual_list)
    
#     @parameterized.expand([
#         ['0010'],
#         ['----'],
#         ['1']
#         ['']
#     ])
#     def test_valid_key(self, input_key):
#         actual = is_valid_key(input_key)
        
#         self.assertTrue(actual)
    
#     @parameterized.expand([
#         ['001a'],
#         ['----_'],
#         ['111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111']
#         ['23']
#     ])
#     def test_not_valid_key(self, input):
#         actual = is_valid_key(input)
#         self.assertFalse(actual)

#     def test_get_mutations(self):
#         pass


# if __name__ == '__main__':
#     unittest2.main()