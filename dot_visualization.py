import sys, os

import numpy as np
np.seterr(divide = 'ignore') 
import numpy.ma as ma
import pandas as pd


import mitosis_manager

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.major.size'] = plt.rcParams["ytick.major.size"] = 6
plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['font.weight'] = plt.rcParams['axes.titleweight'] = plt.rcParams['axes.labelweight'] = 'bold'


savepath = '.'
if len(sys.argv) > 1:
    savepath = sys.argv[1].split('=', maxsplit=1)[1]

os.makedirs(savepath, exist_ok=True)

RESOLUTION = 1000
window_size = 3
lim = 1.5


db = mitosis_manager.Dataset()
table = db.get_tables('dot_score')

### Figure 2 ###
print('Figure 2:', end='\t', flush=False)

df = mitosis_manager.filter_data(table, {'condition':['SMC2','WT'], 
                                            'time':['G2','2.5min','5min','10min','15min','30min'], 
                                            'preferred':True})
df = mitosis_manager.get_dots(df, resolution=RESOLUTION)

savename = f'2_dots__{mitosis_manager.save_suffix(df)}'
if os.path.exists(f'{savepath}/{savename}.pdf'):
    print(f'FILE ALREADY EXISTS {savepath}/{savename}.pdf', end='\n', flush=True)
else:
    fig, ax = plt.subplots(figsize=(0.5+window_size*6+1, 0.5+2*window_size), 
                           nrows=2, ncols=6+1, 
                           sharex='col', sharey='col',
                           constrained_layout=True,
                           gridspec_kw={'width_ratios':[100]*6+[3]},
                          )

    for i, cond in enumerate(['SMC2', 'WT']):
        group = df.query(f"condition == '{cond}'").reset_index(drop=True)

        for j, row in group.iterrows():
            time = row['time']

            pileup = row[f'dots_{RESOLUTION}']
#             pileup = pileup[10:-10,10:-10]
            img = ax[i,j].imshow(np.log2(pileup), cmap='coolwarm', vmin=-lim, vmax=lim)
            
            ax[i,j].set_aspect('equal')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

            if not i:
                ax[i,j].set_title(time)

        ax[i,0].set_ylabel(cond)

        plt.colorbar(img, cax=ax[i,-1], extend='both', ticks=[-lim, -np.round(lim/2, 2), 0, np.round(lim/2, 2), lim])
        
    fig.savefig(f'{savepath}/{savename}.pdf',format='pdf')
    print(f'File saved to {savepath}/{savename}.pdf', end='\n', flush=True)
    
    
savename = f'2_dot_score_trajectory__{mitosis_manager.save_suffix(df)}'
if os.path.exists(f'{savepath}/{savename}.pdf'):
    print(f'FILE ALREADY EXISTS {savepath}/{savename}.pdf', end='\n', flush=True)
else:
    fig, ax = plt.subplots(nrows=1, ncols=2, 
                           figsize=(9,6), 
                           gridspec_kw={'width_ratios':[100,15]})


    time_array = np.array([0, 2.5, 5, 10, 15, 18])
    time_names = np.array(['G2', '2.5min', '5min','10min', '15min', '30min'])
    time_labels = np.array(['G2', '2.5', '5','10', '15', '30'])

    for cond, group in df.groupby('condition'):
            
        times = group['time'].values
        time_mask = np.isin(time_names, times)

        dots = group[f'dot_score_{RESOLUTION}'].values
        dots = (dots - 1)/(dots[0] - 1)
        
        ax[0].plot(time_array[time_mask], dots, 
                     lw=4, markersize=16, marker='o', linestyle='-', 
                     color=mitosis_manager.condition_colors[cond], 
                     label=cond)
        
        ax[1].plot(time_array[time_mask], dots, 
                   lw=4, markersize=16, marker='o', linestyle='-', 
                   color=mitosis_manager.condition_colors[cond], 
                   label=cond)


    ax[0].set_ylabel('Dots')


    for a in ax:
        a.set_xticks(time_array)
        a.set_xticklabels(time_labels, rotation=90)
        a.set_ylim([-0.1, 1.1])

    d = .03
    f1 = 0.5
    f2 = 3.25

    ax[0].legend()
    ax[0].set_xlabel('Time (Minutes)')
    ax[0].set_xlim(-0.5, 15.75)  # outliers only
    ax[0].spines['right'].set_visible(False)


    kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)

    ax[0].plot((1 - f1*d, 1 + f1*d), (-d, +d), lw=2, **kwargs)  
    ax[0].plot((1 - f1*d, 1 + f1*d), (1 - d, 1 + d), lw=2, **kwargs)        

    kwargs.update(transform=ax[1].transAxes)  
    ax[1].plot((-f2*d, +f2*d), (1 - d, 1 + d),  lw=2,**kwargs) 
    ax[1].plot((-f2*d, +f2*d), (-d, +d), lw=2, **kwargs)  

    ax[1].set_xlim(16.25, 18.5)
    ax[1].spines['left'].set_visible(False)
    ax[1].yaxis.set_ticks([])

    fig.tight_layout(w_pad=0.25)
    fig.savefig(f'{savepath}/{savename}.pdf',format='pdf')
    print(f'File saved to {savepath}/{savename}.pdf', end='\n', flush=True)