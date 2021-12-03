import pandas as pd
import math


if __name__ == '__main__':
    data1 = {
        'a':[1,10],
        'b':[1, 10],
        'c':[1, 10]
    }
    
    data2 = {
        'a':[1,10],
        'b':[1, 10],
        'c':[1, 10]
    }
    
    # Create DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Print the output.
    print(df1)
    print('---')
    print(df2)
    print('---')
    df3 = df1['a'] + df2['a']
    print(df3)
    print('---')
    # print(sum(df3['a'], df3['a']))
