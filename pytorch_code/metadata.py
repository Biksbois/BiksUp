
import copy
from icecream import ic
from unittest2 import util
from pretty_printer import print_warning, print_message

ZERO = '0'
ONE = '1'
BOTH = '_'

data_dict =  {
    'key_01':'This is test key 1',
    'key_02':'This is test key 2',
    'key_03':'This is test key 3',
    'key_04':'This is test key 4',
    'key_05':'This is test key 5',
    'key_06':'This is test key 6',
    'key_07':'This is test key 7',
    'key_08':'This is test key 8',
}

VALID_KEY_VALUES = [ZERO, ONE, BOTH]
DEFAULT_BOOL = True
DEFAULT_CHAR = ONE if DEFAULT_BOOL == True else ZERO

def get_default_dict():
    new_dict = dict(data_dict)
    
    for value in new_dict.values():
        value = DEFAULT_BOOL
    
    return new_dict

def introduce_keys(lst):
    msg = f"A new experiment is about to execute.\nThis experiment will execute with {len(lst)} variation(s).\nThese varations include:"
    
    for obj in lst:
        msg += f"\n  - {obj.get_key()}"
    
    msg += "\nEach index in the keys represents the following:\n"
    
    tmp = metadataObj()
    
    msg += tmp.get_keys_meaning()
    
    print_message(msg)


def get_key_count():
    obj = metadataObj()
    return len(obj.meta_dict.keys())

def get_metadata_list(key_list):
    assert type(key_list) == list, "'get_metadata_list' -  This method should recieve a list as input."
    
    metadata_list = []
    key_mutations = []
    key_count = get_key_count()
    
    for key in key_list:
        if is_valid_key(key, key_count):
            key_mutations.extend(get_key_mutations(key))
        else:
            print_warning(f"The following key was not valid: {key}", method_name='get_metadata_list', end_program=True)

    key_mutations = [x.ljust(key_count, DEFAULT_CHAR) for x in key_mutations]
    key_mutations = list(set(key_mutations))
    
    for keymutation in key_mutations:
        metadata_list.append(metadataObj(keymutation))
    
    return metadata_list

def is_valid_key(key, key_count):
    if len(key) > key_count:
        return False
    else:
        for i in key:
            if not i in VALID_KEY_VALUES:
                return False
        return True

def generate_mutations(key):
    ret_list = []
    
    for i in key:
        if i == BOTH:
            if len(ret_list) == 0:
                ret_list.append(ZERO)
                ret_list.append(ONE)
            else:
                l_0 = copy.copy(ret_list)
                l_1 = copy.copy(ret_list)
                
                l_0 = [x + ZERO for x in l_0]
                l_1 = [x + ONE for x in l_1]
                
                ret_list = l_0 + l_1
        else:
            if len(ret_list) == 0:
                ret_list.append(i)
            else:
                ret_list = [x + i for x in ret_list]
    
    return ret_list

def get_key_mutations(key):
    if not BOTH in key:
        return [key]
    else:
        ret_list = []
        
        ret_list = generate_mutations(key)
        
        return ret_list

class metadataObj():
    def __init__(self, key='') -> None:
        self.meta_dict = get_default_dict()
        
        dict_keys = list(self.meta_dict.keys())
        for i in range(len(key)):
            cur_dict_key = dict_keys[i]
            self.meta_dict[cur_dict_key] = True if key[i] == ONE else False
    
    
    def print_key_value(self):
        for key in self.meta_dict.keys():
            print(f"{key} - {self.meta_dict[key]}")
    
    def get_key_count(self):
        return len(self.meta_dict.keys())
    
    def test_case(self, key):
        if key in self.meta_dict:
            return self.meta_dict[key]
        else:
            # TODO: Crash the program
            print("oh no")
        # return key in self.meta_dict and self.meta_dict[key] == True
    
    def get_key(self):
        lst = list(self.meta_dict.values())
        ret = ""
        for s in lst:
            ret += ONE if s == True else ZERO
        return ret
    
    def get_keys_meaning(self):
        msg = ""
        
        for i in range(len(list(self.meta_dict.keys()))):
            msg += f"\n    - {str(i).rjust(3, ZERO)} {list(self.meta_dict.keys())[i]}"
        
        return msg
    
    def __eq__(self, other):
        return self.meta_dict == other.meta_dict


if __name__ == '__main__':
    input = ['0011011', '0011111', '0000', '0001']
    p = get_metadata_list(input)
    
    a = metadataObj()
