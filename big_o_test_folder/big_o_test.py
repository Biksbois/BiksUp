import big_o

def find_max(lst):
    _max = 0
    for i in lst:
        if i > _max:
            _max = i
    return _max

def kwa(q, **kwargs):
    for key, value in kwargs.items():
        print(f"key: {key}, value: {value}")

if __name__ == '__main__':
    # positive_int_generator = lambda n: big_o.datagen.integers(n, 0, 10000)
    # best, others = big_o.big_o(find_max, positive_int_generator, n_repeats=100)
    # print(best)
    
    d = {"a":1, "b":2, "c":3}
    kwa("a", a=1, b=2, c=3)