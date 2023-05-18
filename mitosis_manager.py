import os
import sys
import glob

from collections import defaultdict
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from cooler import Cooler
import h5py
import shelve

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


base_path = '/net/levsha/share/lab/chicken_project/chicken2.0/galGal7b'
storage_path = '/net/levsha/scratch/sameer/chicken/galGal7'
db_path = '/home/sameer/mitosis/galGal7/metadata/galGal7_info'
# deep_library = 'NoSMC3-G2-deep'

condition_colors = {'WT':'#b14e02',
                    
                    'SMC3 +':'#0073e6',
                    'SMC3 -':'#b14e02',
                    
                    'SMC3 --':'#b14e02',
                    'SMC3 +-':'#fd8c35',
                    'SMC3 -+':'#0073e6',
                    'SMC3 ++':'#66b3ff',
                    
                    'SMC3-SMC2':'#913bab',
                    'SMC2':'#c68ed7',
                    
                    'SMC3-CAPH':'#e60000',
                    'CAPH':'#ff6666',
                    
                    'SMC3-CAPH2':'#39ac39',
                    'CAPH2':'#8cd98c',
                     }



def get_chromsizes():
    return pd.read_csv(f'{base_path}/galGal7b_chromsizes.txt', sep='\t')

def get_chromarms():
    
    return pd.read_csv(f'{base_path}/galGal7b_arms.tsv', sep='\t')

def save_suffix(df, column=None): 
    
    if column is not None:
        columns = ['condition', column, 'replicate']
    else:
        columns = ['condition', 'replicate']
    
    return '_'.join(list('-'.join(map(str, properties)) for properties in (df[columns]
                                         .drop_duplicates()
                                         .sort_values(columns)
                                         .itertuples(index=False, name=None))))

class Dataset:
    
    def __init__(self):
        with shelve.open(db_path, flag='r') as db:
            k = list(db.keys())
            k.remove('metadata')
            self.keys = k
            self.metadata = db['metadata']
            
    
    def get_tables(self, keys=None):
        if keys is None:
            return self.metadata
        
        if isinstance(keys, str):
            keys = [keys]
            
        result = self.metadata
        with shelve.open(db_path, flag='r') as db:
            for key in keys:
                assert key in self.keys, "Key not found in database"
                df = db[key]
                
                result = result.merge(df, on='lib_name', how='outer')
        
        return result

    def add_table(self, key, table):
        assert 'lib_name' in table.columns, "Please pass table with lib_names columns in it"
        table_lib_names = table['lib_name'].values
        meta_lib_names = self.metadata['lib_name'].values

        with shelve.open(db_path, flag='w') as db:
            assert key not in self.keys, "Key already exists. If you wish to modify this, please use modify_table() method"
            assert np.all(meta_lib_names == table_lib_names), 'List of libraries does not match those in metadata'
            
            db[key] = table
            self.keys.append(key)
     
    def remove_table(self, key):
        assert key != 'metadata', "'metadata' table should not be deleted."
            
        with shelve.open(db_path, flag='w') as db:
            assert key in self.keys, "Key not found in database"
            
            del db[key]
            self.keys.remove(key)
            
    def modify_table(self, key, new_table):
        assert 'lib_name' in new_table.columns, "Please pass table with lib_names columns in it"
        table_lib_names = new_table['lib_name'].values
        
        with shelve.open(db_path, flag='w') as db:
            assert key in list(db.keys()), "Key not found in database. If you want to add a table, please use add_table() method"
            meta_lib_names = db['metadata']['lib_name'].values
            assert np.all(meta_lib_names == table_lib_names), 'List of libraries does not match those in metadata'
            
            db[key] = new_table
                
    
def filter_data(df, filter_dict):
    for key in filter_dict.keys():
        assert key in df.columns, f'Column named {key} not found in DataFrame'
        
    out_df = df.copy()
    for dict_item in filter_dict.items():
        out_df = find_matches(out_df, dict_item)
    
    return out_df

def find_matches(in_df, dict_item):
    col, val = dict_item
    
    if isinstance(val, bool):
        out_df =  in_df[in_df[col] == val]
    elif isinstance(val, str):
        out_df = in_df[in_df[col] == val]
    else:
        out_df = []
        for item in val:
            out_df.append(find_matches(in_df, (col, item)))
        out_df = pd.concat(out_df)
    
    return out_df


def get_coolers(table, resolution):

    cool_dict = defaultdict(list)   
    for name in table['lib_name']:
        cool_dict['lib_name'].append(name)
        
        filename = f'{name}__galGal7b.mapq_30.1000.mcool'
        if filename in os.listdir(f'{base_path}/coolers'):
            cool_dict[f'cooler_{resolution}'].append(f'{base_path}/coolers/{filename}::/resolutions/{resolution}')
        else:
            cool_dict[f'cooler_{resolution}'].append(np.nan)
    
    df = pd.DataFrame(cool_dict)
    df = table.merge(df, on='lib_name', how='outer')
    return df

def get_pairs(table):

    pairs_dict = defaultdict(list)
    
    for name in table['lib_name']:
        pairs_dict['lib_name'].append(name)
        
        suffix = '__galGal7b.nodups.pairs.gz'
        if len(name.split('-')[-1]) == 5:
            suffix = '__galGal7b.galGal7b.nodups.pairs.gz'
            
        filename = f'{name}{suffix}'
        if filename in os.listdir(f'{base_path}/pairs'):
            pairs_dict['pairs'].append(f'{base_path}/pairs/{filename}')
        else:
            pairs_dict['pairs'].append(np.nan)
    
    df = pd.DataFrame(pairs_dict)
    df = table.copy(deep=True).merge(df, on='lib_name', how='outer')
    return df


def get_contact_scalings(table, resolution=1000):
    
    scale_path = f'{storage_path}/contact_scalings/{resolution}' 
    
    scale_dict = defaultdict(list)
    for name in table['lib_name']:
        scale_dict['lib_name'].append(name)
               
        try:
            df = pd.read_csv(f'{scale_path}/{name}.txt', sep='\t')
        except:
            df = None
        scale_dict[f'Ps_{resolution}'].append(df)

    df = pd.DataFrame(scale_dict)
    df = table.copy(deep=True).merge(df, on='lib_name', how='outer')

    return df

def save_contact_scaling(Ps_table, lib_name, resolution):
    
    scale_path = f'{storage_path}/contact_scalings/{resolution}' 
    os.makedirs(scale_path, exist_ok=True)
    
    if os.path.exists(f'{scale_path}/{lib_name}.txt'):
        raise FileExistsError(f'File already exists at: {scale_path}/{lib_name}.txt')
        
    Ps_table.to_csv(f'{scale_path}/{lib_name}.txt', sep='\t', header=True, index=False)
    


def get_saddles(table, resolution):
    
    saddle_path = f'{storage_path}/saddles/{resolution}'

    saddle_dict = defaultdict(list)
    for name in table['lib_name']:
        saddle_dict['lib_name'].append(name)

        if f'{name}.npy' in os.listdir(saddle_path):
            saddle = np.load(f'{saddle_path}/{name}.npy')
            saddle_sums, saddle_counts = saddle[:,:,0], saddle[:,:,1]
        else:
            saddle_sums, saddle_counts = None, None
            
        saddle_dict[f'saddle_sum_{resolution}'].append(saddle_sums)
        saddle_dict[f'saddle_count_{resolution}'].append(saddle_counts)

    df = pd.DataFrame(saddle_dict)
    df = table.copy(deep=True).merge(df, on='lib_name', how='outer')
    
    return df

def save_saddle(saddle_sums, saddle_counts, lib_name, resolution):
    
    saddle_path = f'{storage_path}/saddles/{resolution}'
    os.makedirs(saddle_path, exist_ok=True)
    
    if os.path.exists(f'{saddle_path}/{lib_name}.npy'):
        raise FileExistsError(f'File already exists at: {saddle_path}/{lib_name}.npy')
    
    np.save(f'{saddle_path}/{lib_name}.npy', np.dstack((saddle_sums,saddle_counts)))
    
    
def get_dots(table, resolution):
        
    dot_path = f'{storage_path}/dots/{resolution}'
    os.makedirs(dot_path, exist_ok=True)

    dot_dict = defaultdict(list)
    for name in table['lib_name']:
        dot_dict['lib_name'].append(name)  
        
        if f'{name}.npy' in os.listdir(dot_path):
            arr = np.load(f'{dot_path}/{name}.npy')
        else:
            arr = None
        
        dot_dict[f'dots_{resolution}'].append(arr)
        
    df = pd.DataFrame(dot_dict)
    df = table.copy(deep=True).merge(df, on='lib_name', how='outer')
    
    return df

def save_dots(pileup, lib_name, resolution):
    
    dot_path = f'{storage_path}/dots/{resolution}'
    os.makedirs(dot_path, exist_ok=True)
    
    if os.path.exists(f'{dot_path}/{lib_name}.npy'):
        raise FileExistsError(f'File already exists at: {dot_path}/{lib_name}.npy')
    
    np.save(f'{dot_path}/{lib_name}.npy', pileup)

class Pileup():
    
    def __init__(self, path, file_name):
        if path[-1] != '/':
            path = f'{path}/'
        self.path = path
        self.name = file_name
                
    def load(self):

        if self.name not in os.listdir(self.path):
            return np.nan
        else:
            return np.load(f'{self.path}{self.name}')

# def get_dot_calls(data, anchors=False, anchor_rad=None):
    
#     if anchors:
#         assert isinstance(anchor_rad, int)

#     names = data['lib_name'].values
    
#     dot_dict = defaultdict(list)#{'lib_name':[], f'insulation_{res}':[]}
#     for i, name in enumerate(names):
#         dot_dict['lib_name'].append(name)
        
#         if name in os.listdir(dot_paths[0]):
#             files  = glob.glob(f'{dot_paths[0]}/{name}/combineddots/*.postproc')
#             if len(files) == 1:
#                 df = pd.read_csv(files[0], sep='\t')
#                 dot_dict['dot_list'].append(df)
#             else:
#                 print(f'Searching directory: {dot_paths[0]}/{name}/combinneddots')
#                 print(f'Either zero or multiple dot files found associated with name: {name}')
#                 dot_dict[f'anchors_{anchor_rad}'].append(np.nan)

#             if anchors:
#                 files  = glob.glob(f'{dot_paths[0]}/{name}/combineddots/*.postproc.anchors_{anchor_rad}')
#                 if len(files) == 1:
#                     df = pd.read_csv(files[0], sep='\t')
#                     dot_dict[f'anchors_{anchor_rad}'].append(df)
#                 else:
#                     print(f'Searching directory: {dot_paths[0]}/{name}/combinneddots')
#                     print(f'Either zero or multiple anchor files found associated with name: {name} and cluster radius: {anchor_rad}')
#                     dot_dict[f'anchors_{anchor_rad}'].append(np.nan)
                
#         elif name in os.listdir(dot_paths[1]):
#             files  = glob.glob(f'{dot_paths[1]}/{name}/combineddots/*.postproc')
#             if len(files) == 1:
#                 df = pd.read_csv(files[0], sep='\t')
#                 dot_dict['dot_list'].append(df)
#             else:
#                 print(f'Searching directory: {dot_paths[1]}/{name}/combinneddots')
#                 print(f'Either zero or multiple dot files found associated with name: {name}')
#                 dot_dict[f'anchors_{anchor_rad}'].append(np.nan)

#             if anchors:
#                 files  = glob.glob(f'{dot_paths[1]}/{name}/combineddots/*.postproc.anchors_{anchor_rad}')
#                 if len(files) == 1:
#                     df = pd.read_csv(files[0], sep='\t')
#                     dot_dict[f'anchors_{anchor_rad}'].append(df)
#                 else:
#                     print(f'Searching directory: {dot_paths[1]}/{name}/combinneddots')
#                     print(f'Either zero or multiple anchor files found associated with name: {name} and cluster radius: {anchor_rad}')
#                     dot_dict[f'anchors_{anchor_rad}'].append(np.nan)
                
#         else:
#             print(f'Could not find library : {name}')
#             dot_dict[f'dot_list'].append(np.nan)
#             if anchors:
#                 dot_dict[f'anchors_{anchor_rad}'].append(np.nan)

#     df = pd.DataFrame(dot_dict)
#     df = data.copy(deep=True).merge(df, on='lib_name', how='outer')
    
#     return df

# def get_tads(data, tad_list, res):
        
#     names = data['lib_name'].values
    
#     tad_dict = {'lib_name':[], f'tads_{res}':[]}
#     for i, name in enumerate(names):
#         tad_dict['lib_name'].append(name)
#         tad_path = analysis_path+f'pileups/tads/{tad_list}/{res}/'
#         tad_dict[f'tads_{res}'].append(Pileup(tad_path, f'{name}.npy'))

#     df = pd.DataFrame(tad_dict)
#     df = data.copy(deep=True).merge(df, on='lib_name', how='outer')
    
#     return df

