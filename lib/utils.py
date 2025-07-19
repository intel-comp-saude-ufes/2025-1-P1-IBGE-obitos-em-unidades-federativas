import os
import pandas as pd

def read_processed_data_frame(path ='./', sep = ','):
    verify_path(path)
    df = pd.read_csv(path, sep=sep)
    return df

def verify_path(path : str):
    if not os.path.exists(path) : raise Exception("path does not exists")
    return


'''
Renzo : 0 a 14 e 15 a 29
Saick : 60 ou mais e 70 ou mais
'''
def filter_columns_by_age_range(data : pd.DataFrame, range_tup = (0, 14)):
    
    possible_ranges = ['0 a 14', '15 a 29', '30 a 44', '45 a 59', '60 anos', '60 a 69', '70 anos']
    
    y1, y2 = range_tup
    
    if y2 == -1:
        range_str = f'{y1} anos'
    else :
        range_str = f'{y1} a {y2}'
    
    if range_str not in possible_ranges:
        _error_str = f'the age range {range_str} is not valid'
        raise Exception(_error_str)

    selected_columns = filter(lambda _c : range_str in _c, data.columns)
    selected_columns = list(selected_columns)
    
    data_selected = data[selected_columns].copy()
    return data_selected

def print_dict(m : dict):
    
    for k,v in m.items():
        print('\t', k)
        try:
            v = iter(v)
        except:
            v = iter([v])
        for _e in v:
            print('\t\t', _e) 