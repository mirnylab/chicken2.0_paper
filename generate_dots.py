import collections
import glob, os
import multiprocessing as mp
import time

import numpy as np
import numpy.ma as ma
import pandas as pd
np.seterr(invalid='ignore')

import matplotlib
import matplotlib.pyplot as plt

import cooler
import cooltools
import cooltools.lib.plotting
import bioframe

import mitosis_manager


# resolution = 5_000
# flank_bp = 20*resolution

resolution = 1_000
flank_bp = 50*resolution


galGal7b_arms = mitosis_manager.get_chromarms()
MIN_ARM_SIZE = int(3e7)
view_df = galGal7b_arms.query(f'end - start >= {MIN_ARM_SIZE}').reset_index(drop=True)

dot_calls = pd.read_csv('dot_list.txt', sep='\t')
mask = dot_calls['region1'].apply(lambda x: x in view_df['name'].unique())
dot_calls = dot_calls.loc[mask].reset_index(drop=True)

### Constructing Dot Pileups
db = mitosis_manager.Dataset()
table = db.get_tables()
# df = mitosis_manager.filter_data(table, {'condition':['WT','SMC2'], 'preferred':True})
df = mitosis_manager.filter_data(table, {'condition':['WT','SMC2','CAPH','CAPH2'], 'preferred':True})

df = mitosis_manager.get_coolers(df, resolution=resolution)
df = mitosis_manager.get_contact_scalings(df, resolution=resolution)
df = mitosis_manager.get_dots(df, resolution=resolution)

dot_scores = collections.defaultdict(list)
for _, row in df.iterrows():
    lib_name = row['lib_name']
    
    print(f'{lib_name}:\t', end='\t', flush=True)
    
    pileup = row[f'dots_{resolution}']
    
    clr = cooler.Cooler(row[f'cooler_{resolution}'])
    expected = row[f'Ps_{resolution}']
    
    if isinstance(pileup, np.ndarray):
        print('Already computed', end='\n', flush=True)
        
    elif expected is None:
        print('Expected file not found...', end='\t', flush=True)
        time.sleep(0.5)
        print('Skipping', end='\n', flush=True)
        continue

    else:
        print(f'Computing {resolution//1000} kb pileup...', end='\t', flush=True)
        mask = expected['region1'].apply(lambda s: s in view_df['name'].unique())
        expected = expected.loc[mask]

        try:
            stack = cooltools.pileup(clr, 
                                     dot_calls, 
                                     view_df=view_df, 
                                     expected_df=expected, 
                                     flank=flank_bp,
                                     nproc=10
                                    )
        except Exception as e:
            print(f'Error!\n{e}', end='\n', flush=True)
            continue

        pileup = np.nanmean(stack, axis=2)

        mitosis_manager.save_dots(pileup, row['lib_name'], resolution)
        print('DONE!', end='\n', flush=True)
        
    size = pileup.shape[0]
    mid = size//2
    half_width = 1
    slicing = slice(mid - half_width, mid + half_width + 1)
    score = np.mean(pileup[slicing, slicing])

    dot_scores['lib_name'].append(lib_name)
    dot_scores[f'dot_score_{resolution}'].append(score)
    
dot_scores = pd.DataFrame(dot_scores)
dot_scores = table[['lib_name']].merge(dot_scores, how='outer')
db.modify_table('dot_score', dot_scores)