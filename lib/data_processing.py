import pandas as pd
from tqdm import tqdm
import os
from lib.colors import *
from functools import reduce
from pathlib import Path

def read_file(file_path, year_str):
  df = None
  status = 'ok'

  ## tenta ler a 'sheet'
  try:
    df = pd.read_excel(file_path, sheet_name=year_str)
  except:
    status = 'err'
    df = None

  ## se não conseguir, le normalmente
  if df is None:
    df = pd.read_excel(file_path)

  return (df, status)

def get_dir_content(root='./'):
  may_files = (map(lambda x : os.path.join(root, x), os.listdir(root)))
  folders = []
  files = []
  for may_file in may_files:
    if os.path.isfile(may_file) : files.append(may_file)
    else: folders.append(may_file)
  return (folders, files)

# Primeira linha tem ter o título
# Tem que ter a linha com Brasil

def process_data_frame(df):
  #cols = [f'Unnamed: {i}' for  i in range(len(df.columns))]
  #df.columns = cols

  # Estados
  states = set([
    "Brasil","Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará",
    "Distrito Federal", "Espírito Santo", "Goiás", "Maranhão", "Mato Grosso",
    "Mato Grosso do Sul", "Minas Gerais", "Pará", "Paraíba", "Paraná",
    "Pernambuco", "Piauí", "Rio de Janeiro", "Rio Grande do Norte",
    "Rio Grande do Sul", "Rondônia", "Roraima", "Santa Catarina",
    "São Paulo", "Sergipe", "Tocantins"
  ])

  # Removendo linhas nulas
  nan_rows  = df.apply(lambda _row: _row.isna().sum() == len(_row), axis=1)
  df[~nan_rows].reset_index(drop=True, inplace=True)

  temp_columns = ['Estados'] + list(df.columns.values[:-1])
  df.columns = temp_columns

  index_brasil = df[df['Estados'] == 'Brasil'].index[0]
  df_header = df[:index_brasil] ## até antes de brasil

  df_header = df_header.ffill(axis=0).ffill(axis=1)

  def build_col_name(col):
      col = col.dropna().astype(str).str.strip()
      parts = []
      for level in col:
          # Evita adicionar nível igual ao último adicionado
          if not parts or parts[-1] != level:
              parts.append(level)
      return '_'.join(parts)

  mask_states = df['Estados'].apply(lambda x : x in states)
  df = df.loc[mask_states]

  rows_to_drop = []
  index_sao_paulo = df[df['Estados'] == 'São Paulo'].index

  if index_sao_paulo.size > 1:
    rows_to_drop.append(index_sao_paulo[1])

  index_rio_janeiro = df[df['Estados'] == 'Rio de Janeiro'].index

  if index_rio_janeiro.size > 1:
    rows_to_drop.append(index_rio_janeiro[1])

  df = df.drop(rows_to_drop) # jogando fora segunda ocorrencia de sao paulo e rio de janeiro
  df.reset_index(drop=True, inplace=True)

  new_column_names = df_header.apply(build_col_name, axis=0)
  df.columns = new_column_names.tolist()

  return df


def process_all_years(path='./', prefix_to_save = 'result'):

  folders = ['1-uteis', '2-uteis','3-uteis', '4-uteis', '5-uteis', '5-uteis-parte-1']
  years = list(map(lambda x : str(x), [2023]))
  
  ## ignorar arquivos já processados
  file_filter = lambda _file : ('done' not in _file) and ('Tabela' in _file)

  grades_regioes_str = 'Grandes Regiões'
  done_df_list = []

  for y_str in years:

      df_list = []
      test_list = []

      for folder in folders:

        content = get_dir_content(os.path.join(path, folder))
        files = sorted(content[1])
        
        file_tqdm = tqdm(files)
        for file in file_tqdm:
            file_tqdm.set_description_str(f"File Name : {file}")
        
            if not file_filter(file): 
                print('',f'{COLOR_YELLOW}[WARNING]{COLOR_RESET} the file `{file}` was ignored.')
                continue

            df, status = read_file(file, y_str)
            df = process_data_frame(df)

            file_path = file[0:file.index('.xls')]
            ext = file[file.index('.xls'):]
            base_name = Path(file).stem

            ## colocando nome da tabela na coluna
            df.columns = [f'{base_name}_{_c}' for _c in list(df.columns) ]
            df.columns = list(map(lambda x : grades_regioes_str if grades_regioes_str in x else x, list(df.columns)))

            ## caso não tenha a 'sheet' do ano, tem uma coluna daquele ano
            if status == 'err':
                year_columns = [_c for _c in df.columns if y_str in _c] + [grades_regioes_str]
                df = df[year_columns]
                
            df_list.append(df)
          
      df_final = reduce(lambda left, right: pd.merge(left, right, on= grades_regioes_str), df_list)
      df_save_name = f'{prefix_to_save}_{y_str} - done.csv'
      print(f'{COLOR_BLUE}[INFO]{COLOR_RESET} Saving the table `{df_save_name}`')
      df_final.to_csv(df_save_name, index=False)
      done_df_list.append(df_final)

  return done_df_list