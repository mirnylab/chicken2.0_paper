import sys, os
import multiprocessing as mp

import numpy as np
import numpy.ma as ma
import pandas as pd

np.seterr(invalid='ignore')
np.random.seed(31415)

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import networkx as nx
from scipy.spatial import cKDTree

import matplotlib
import matplotlib.pyplot as plt

import cooler
import cooltools
import bioframe

import mitosis_manager


RESOLUTIONS = [5_000, 10_000]
CHUNKSIZE = int(1e7)

MAX_LOCI_SEPARATION = 1_000_000
CLUSTERING_RADIUS = 30_000
FALSE_DISCOVERY_RATE = 0.15

merged_cooler_name = 'dot-merged-library__galGal7b.mapq_30.mcool'

nproc = 1
if len(sys.argv) > 1:
    nproc = int(sys.argv[1].split('=', maxsplit=1)[1])

galGal7b_arms = mitosis_manager.get_chromarms()

print('PART I: EXTRACTING DOT LISTS AT 5KB AND 10KB\n', end='\n', flush=True)
db = mitosis_manager.Dataset()
table = db.get_tables()

subset = []
subset.append(
    mitosis_manager.filter_data(table, {'condition':['SMC3','SMC3-CAPH','SMC3-CAPH2'],
                                          'A':False, 
                                          'time':'G2', 'preferred':True}
                                 )
            )

subset.append(
    mitosis_manager.filter_data(table, {'condition':'SMC3',
                                          'A':['Aa','aa'], 
                                          'time':'G2', 'preferred':True}
                                 )
            )

subset.append(
    mitosis_manager.filter_data(table, {'condition':['CAPH','CAPH2','SMC2','WT'], 
                                          'time':'G2','preferred':True}
                                 )
            )

subset = pd.concat(subset)
   
    
dot_sets = {}
all_dots = []
for resolution in RESOLUTIONS:
    print(f'RESOLUTION:\t{resolution//1000}kb\n', end='\n', flush=True)
    
    try:
        clr = cooler.Cooler(f'{merged_cooler_name}::/resolutions/{resolution}')
        print(f'Merged cooler already found at {merged_cooler_name}::/resolutions/{resolution}\n', end='\n', flush=True)
    except:
        print(f'\tCreating merged cooler...', end='\t', flush=False)
        df = mitosis_manager.get_coolers(subset, resolution=resolution)
        cooler.merge_coolers(
            f'{merged_cooler_name}::/resolutions/{resolution}', 
            df[f'cooler_{resolution}'].values, 
            CHUNKSIZE, 
            mode='a'
        )

        clr = cooler.Cooler(f'{merged_cooler_name}::/resolutions/{resolution}')
        
        with mp.Pool(nproc) as p:
            cooler.balance_cooler(clr, CHUNKSIZE, p.map,
                                  min_nnz=10, min_count=0, mad_max=8, 
                                  cis_only=False, trans_only=False, 
                                  ignore_diags=2, store=True, store_name='weight'
                                 )
        print(f'DONE\n\tSaved to {merged_cooler_name}::/resolutions/{resolution}\n', end='\n', flush=True)
    
    print(f'\tComputing Expected...', end='\t', flush=False)
    expected = cooltools.expected_cis(clr,
                                      view_df=galGal7b_arms,
                                      nproc=nproc,
                                     )
    print(f'DONE\n', end='\n', flush=True)

    print(f'\tCalling Dots...', end='\t', flush=False)
    dot_sets[resolution] = cooltools.dots(
                                        clr,
                                        expected=expected,
                                        view_df=galGal7b_arms,
                                        kernels=None,
                                        max_loci_separation=MAX_LOCI_SEPARATION,
                                        clustering_radius=CLUSTERING_RADIUS,
                                        tile_size=5_000_000,
                                        nproc=nproc,
                                        lambda_bin_fdr=FALSE_DISCOVERY_RATE,
                                        max_nans_tolerated=2
                                    )
    
    print(f'DONE\n', end='\n', flush=True)

    dots = dot_sets[resolution].copy(deep=True).reset_index()
    dots['pos1'] = (dots['start1'] + dots['end1'])//2
    dots['pos2'] = (dots['start2'] + dots['end2'])//2
    dots = dots[['region','pos1','pos2','index']]
    dots['source'] = resolution
    all_dots.append(dots)

all_dots = pd.concat(all_dots).sort_values(['region','pos1','pos2'])

print('PART II: MERGING DOT LISTS TOGETHER...\n', end='\n', flush=True)
combined_dots = []
for region, group in all_dots.groupby('region'):
    group = group.reset_index(drop=True)
    
    G = nx.Graph()
    G.add_nodes_from(np.arange(group.shape[0]))
    

    X = group[['pos1','pos2']].values
    kdt = cKDTree(X)    
    G.add_edges_from(kdt.query_pairs(CLUSTERING_RADIUS))
    
    indices = []
    for cluster in nx.connected_components(G):
        if len(cluster) == 1:
            indices += list(cluster)
        else:
            indices.append(np.random.choice(list(cluster)))
            
    reduced_group = group.loc[np.array(indices)]
    
    for res, df in reduced_group.groupby('source'):
        combined_dots.append(dot_sets[res].loc[df['index'].values])

combined_dots = pd.concat(combined_dots).sort_values(['region','start1','start2']).reset_index(drop=True)
combined_dots.to_csv('dot_list.txt', sep='\t', header=True, index=False)

print(f'DONE\nSaved to dot_list.txt', end='\n', flush=True)