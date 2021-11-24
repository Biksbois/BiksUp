import statistics

def average(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return sum(lst) / len(lst)

def parse_keys(keys):
    key_str = ""
    
    for key in keys:
        key_str += f"\n    - {key}"
    
    return key_str

# def parse_input_key(keys):
#     key_str = ""
    
#     for key in keys:
#         key_str += f" {key}"
    
#     return key_str

# def create_introduce_file(folder_name, parsed_keys, dataset, root_name):
#     path = os.path.join(root_name, folder_name)
#     dataframes = []
#     keys = []
    
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         keys.append(get_cur_key(file, parsed_keys))
#         dataframes.append(pd.read_csv(file_path))

#     file_content = ("The experiment just ran with the following parameters:\n"
#                     f"  - Iterations: {iterations}\n"
#                     f"  - Keys: {parse_keys(keys)}\n"
#                     f"  - Dataset: {dataset}\n\n"
#                     "The keys are as follows:\n"
#                     f"{key_str}\n\n"
#                     "If you wish to run this experiment again, run the following string:\n"
#                     f"  - python main.py --iterations {iterations} --keys {parse_input_key(keys)} --dataset {dataset}")
    
#     save_path = os.path.join(path, "info.txt")
    
#     f = open(save_path, "w")
#     f.write(file_content)
#     f.close()

def std(lst):
    if len(lst) == 1:
        return 0
    else:
        return statistics.stdev(lst)

def avg_and_std(lst):
    return average(lst), std(lst)

if __name__ == '__main__':
    lst = [1,2,3]
    a, s = avg_and_std(lst)
    print(a)
    print(type(a))
    
    print(s)
    print(type(s))
    
