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
    
