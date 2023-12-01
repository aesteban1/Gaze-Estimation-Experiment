import os
from glob import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
from scipy.io import loadmat # for loading mat files
from tqdm import tqdm_notebook


# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

cwd = os.getcwd()
root_mpi_dir = os.path.join(cwd, 'Data\MPIIGaze')
Data_mpi_dir = os.path.join(root_mpi_dir, 'Data')
img_dir = os.path.join(Data_mpi_dir, 'Original')
ann_dir = os.path.join(root_mpi_dir, 'Annotation Subset')

def read_annot(in_path):
    r_dir = os.path.splitext(os.path.basename(in_path))[0]
    c_df = pd.read_table(in_path, header = None, sep = ' ')
    c_df.columns = ['path' if i<0 else ('x{}'.format(i//2) if i % 2 == 0 else 'y{}'.format(i//2)) for i, x in enumerate(c_df.columns, -1)]
    c_df['path'] = c_df['path'].map(lambda x: os.path.join(img_dir, r_dir, x))
    c_df['group'] = r_dir
    c_df['exists'] = c_df['path'].map(os.path.exists)
    return c_df

all_annot_df = pd.concat([read_annot(c_path) for c_path in glob(os.path.join(ann_dir, '*'))], ignore_index=True)
print(all_annot_df.shape[0], 'annotations')
print('Missing %2.2f%%' % (100-100*all_annot_df['exists'].mean()))
all_annot_df = all_annot_df[all_annot_df['exists']].drop('exists', 1)

all_annot_df.sample(3)
group_view = all_annot_df.groupby('group').apply(lambda x: x.sample(2)).reset_index(drop = True)
fig, m_axs = plt.subplots(2, 3, figsize = (30, 10))
for (_, c_row), c_ax in zip(group_view.iterrows(), m_axs.flatten()):
    c_img = imread(c_row['path'])
    c_ax.imshow(c_img)
    for i in range(7):
        c_ax.plot(c_row['x{}'.format(i)], c_row['y{}'.format(i)], 's', label = 'xy{}'.format(i))
    c_ax.legend()
    c_ax.set_title('{group}'.format(**c_row))
plt.show()