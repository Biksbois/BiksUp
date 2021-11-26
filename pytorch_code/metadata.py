
import copy
from global_items import LAST_TXT, TRUE_LIST
from biksLog import get_logger
import sys

log = get_logger()

ZERO = '0'
ONE = '1'
BOTH = '_'

ILLIGAL_KEYS = ['1_0', '111', '_1_1']

data_dict =  {
    'nonhybrid':'A combination between local and global embedding is utilized',
    'attention':'Is attention utilized',
    'local':'Only local embedding is utilized',
    'uniform_attention' : 'The attention is uniformly distributed between all items in a session',
    'reset_GRU_weights' : 'Is the GRU weights and bias is utilised when updating the reset gate',
    'update_GRU_weights' : 'Is the GRU weights and bias is utilised when updating the update gate',
    'newgate_GRU_weights' : 'Is the GRU weights and bias is utilised when updating the new gate',
    'reset_sigmoid' : 'Is the sigmoid activation function is utilized when calculating the reset gate',
    'input_sigmoid' : 'Is the sigmoid activation function is utilized when calculating the input gate',
    'newgate_tahn' : 'Is the tahn activation function is utilized when calculating the new gate',
}

VALID_KEY_VALUES = [ZERO, ONE, BOTH]
DEFAULT_BOOL = True
DEFAULT_CHAR = ONE if DEFAULT_BOOL == True else ZERO

def get_default_dict():
    new_dict = dict(data_dict)
    
    for value in new_dict.values():
        value = DEFAULT_BOOL
    
    return new_dict

def get_data_dict():
    return data_dict

def get_key_count():
    obj = metadataObj()
    return len(obj.meta_dict.keys())

def save_mutations(key_mutations):
    if len(key_mutations) > 0:
        last_str = key_mutations[0]
        
        for m in key_mutations[1:]:
            last_str += "," + m
        
        try:
            with open(LAST_TXT, 'a') as fd:
                fd.write(f'{last_str}\n')
        except Exception as e:
            log.exception(f"Unabel to add string to last.txt. {e}")
            sys.exit()

def parse_keylist(key_list, is_runlast=False, do_log=True):
    assert type(key_list) == list, "'get_metadata_list' -  This method should recieve a list as input."
    
    metadata_list = []
    key_mutations = []
    key_count = get_key_count()
    
    for key in key_list:
        if is_valid_key(key, key_count, do_log=do_log):
            key_mutations.extend(get_key_mutations(key))
        else:
            sys.exit()

    key_mutations = [x.ljust(key_count, DEFAULT_CHAR) for x in key_mutations]
    key_mutations = list(set(key_mutations))
    key_mutations = [key for key in key_mutations if not contains_illigal_key(key, do_log=do_log)]
    
    
    if not is_runlast:
        save_mutations(key_mutations)
    
    for keymutation in key_mutations:
        metadata_list.append(metadataObj(keymutation))
    
    return metadata_list

def parse_runall():
    ret_str = ''
    keycount = get_key_count()
    
    ret_str = ret_str.ljust(keycount, '_')
    
    return parse_keylist([ret_str])

def parse_runlast():
    try:
        a_file = open(LAST_TXT, "r")
        lines = a_file.readlines()
        if len(lines) > 0:
            last_line = lines[-1].split(',')
            last_line = [x.strip() for x in last_line]
            return parse_keylist(last_line, is_runlast=True)
        else:
            log.warning("No record of previous runs exists. The next run will be default true list")
            return parse_keylist(['0'] , is_runlast=True)
    except Exception as e:
        log.exception(f"Unable to read last key from '{LAST_TXT}'")
        sys.exit()

def get_metadata_list(key_list, runall, runlast, do_log=True):
    if runall in TRUE_LIST:
        return parse_runall()
    elif runlast in TRUE_LIST:
        return parse_runlast()
    else:
        return parse_keylist(key_list, do_log=do_log)

def contains_illigal_key(key, do_log=True):
    illigal_keys = []
    
    for ill in ILLIGAL_KEYS:
        illigal_keys.extend(get_key_mutations(ill))
    
    for illigal in illigal_keys:
        if key[:len(illigal)] == illigal:
            if do_log:
                log.warning(f"The following input key contains the invalid start '{illigal}'. This key will not be added, but the program will continue: '{key}'")
            return True
    return False

def is_valid_key(key, key_count, do_log=True):
    if len(key) > key_count:
        if do_log:
            log.warning(f"The following input key is too long. Keylength = {len(key)} > maxlenght = {key_count}: '{key}'")
        return False
    else:
        for i in key:
            if not i in VALID_KEY_VALUES:
                if do_log:
                    log.warning(f"The following input key contains illigal character. Can only be '{ZERO}', '{ONE}' and '{BOTH}' and not '{i}': '{key}'")
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
            log.exception(f"The following key is not in self.meta_dict: {key}")
            sys.exit()
    
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
    
    def use_attention(self):
        if self.use_weighted_attention() and self.test_case("attention"):
            log.exception(f"Invalid key. Both attention and weighted attention is enabled - {self.get_key()}")
            sys.exit()
        return self.test_case("attention")

    def use_nonhybrid(self):
        return self.test_case("nonhybrid")
    
    def use_weighted_attention(self):
        return self.test_case("uniform_attention")
    
    def use_local(self):
        return self.test_case("local")
    
    def use_GRU_weights(self):
        return self.test_case("GRU_weights")
    
    def use_reset_GRU_weights(self):
        return self.test_case("reset_GRU_weights")
    
    def use_update_GRU_weights(self):
        return self.test_case("update_GRU_weights")
    
    def use_newgate_GRU_weights(self):
        return self.test_case("newgate_GRU_weights")
    
    def use_input_sigmoid(self):
        return self.test_case('input_sigmoid')
    
    def use_reset_sigmoid(self):
        return self.test_case('reset_sigmoid')

    def use_newgate_tahn(self):
        return self.test_case('newgate_tahn')
    
    def count_active_weights(self):
        ret = 0
        if self.use_reset_GRU_weights():
            ret += 1
        if self.use_update_GRU_weights():
            ret += 1
        if self.use_newgate_GRU_weights():
            ret += 1
        return ret
    
    def __eq__(self, other):
        return self.meta_dict == other.meta_dict


if __name__ == '__main__':
    input = ['0011011', '0011111', '0000', '0001']
    p = get_metadata_list(input)
    
    a = metadataObj()
