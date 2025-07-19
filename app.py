import pandas as pd
from tqdm import tqdm

from lib.analysis import plot_silhouette_scores, apply_robust_scaling, pca_analysis
from lib.utils import read_processed_data_frame, filter_columns_by_age_range, print_dict
from lib.data_processing import process_all_years

def generate_silhouette_for_age_ranges(data : pd.DataFrame):    
  ranges = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 69), (60, -1), (70, -1)]

  for rg in ranges:
    df_filtered = filter_columns_by_age_range(data, rg)
    df_filtered_scaled = apply_robust_scaling(df_filtered)    
    
    plot_silhouette_scores(df_filtered_scaled, range_k= range(2,20), img_path=f'./range{rg}.png', title_suffix=f'range{rg}')

def pca_analysis_by_age_ranges(data : pd.DataFrame):
  
  ranges = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 69), (60, -1), (70, -1)]
  
  for rg in ranges:
    print(f'doing range{rg}')
    df_filtered = filter_columns_by_age_range(data, rg)
    print(f'number of columns after range filtering: {len(df_filtered.columns)}')
    
    df_filtered_scaled = apply_robust_scaling(df_filtered)
    important_features = pca_analysis(df_filtered_scaled, data.columns, verbose=True)
    
    for k, v in important_features.items():
      print(k)
      print(f'\t{v}')
      
def remove_column(data : pd.DataFrame, column = ''):
  if not len(column) : return data
  if column not in data.columns:
    print(f'the column {column} does not belong to the dataframe')
    return data
  return data.drop( labels=[column] ,axis=1)

def remove_brasil_row_and_bug_column(data : pd.DataFrame):
  data = data[data['Grandes Regiões'] != 'Brasil']
  data = data.dropna(axis=1, how='all') ## columns that are copies of another ones but with somehow all entries equal to zero.
  return data

def remove_death_related_columns(data : pd.DataFrame, inplace = False)  :
  death_substr = 'Obitos_Geo'
  removed_columns = list(filter(lambda x : death_substr in x, data.columns))
    
  if inplace:
    data.drop(labels=removed_columns, axis=1, inplace=inplace)
    return data
  return data.drop(labels=removed_columns, axis=1)

if __name__ == '__main__':

  process_all_years('./data')


  if False:
      
    df = read_processed_data_frame('all_2022.csv')

    data = remove_brasil_row_and_bug_column(df)
    data = remove_column(data, 'Grandes Regiões')
    data = remove_death_related_columns(data, inplace=True)
    
    feature_columns = data.columns
    X_scaled = apply_robust_scaling(data)
    pca_analysis(X_scaled, feature_columns, verbose=True)
  
  #pca_analysis_by_age_ranges(df)
  

  #generate_silhouette_for_age_ranges(df)
  #df_without_state = remove_column(df, 'Estados_X')
  #pca_analysis_by_age_ranges(df_without_state)
  #process_all_years(path='./data')
  
  
  
  