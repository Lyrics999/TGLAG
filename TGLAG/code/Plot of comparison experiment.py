import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

df_compare = pd.read_csv('.RMSE-cnn-gcn.csv')

df_compare1 = df_compare[(df_compare['model'] == 'CNN-GRU-luong attention')|(df_compare['model'] == 'T-GCN-Luong Attention')]

from matplotlib.colors import to_hex
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
rc = {"weight":600}
sns.set(context='notebook', style='ticks', font_scale=1.3,rc={"font.weight":600})
hex_color_1 = to_hex("#403990", keep_alpha=False)
hex_color_2 = to_hex("#6f8278", keep_alpha=False)
sns.pointplot(x='station',y='mae',hue='model',data=df_compare1,ax=ax1,markers=['o','o'],linestyles=['-','-'], palette={"T-GCN-Self Attenion": hex_color_1, "T-GCN-Luong Attention": hex_color_2})
max_y_value = df_compare1['mae'].max()
ax1.set_ylim(1, max_y_value+0.6)
ax1.set_ylabel('MAE',rc)
ax1.set_xlabel('')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
fig.savefig(".rmsecnn.png",dpi=400)

#hex_color_1 = to_hex("#b36a6f", keep_alpha=False)
#hex_color_2 = to_hex("#6f8278", keep_alpha=False)

#hex_color_1 = to_hex("#403990", keep_alpha=False)
#hex_color_2 = to_hex("#d86967", keep_alpha=False)



