import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import math
import os
import shutil
import tarfile
import utils
import pickle
import fire
import time
import tensorflow as tf
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())




class CropDataset(tf.data.Dataset):
    
    def __new__(cls,df:pd.DataFrame,image_size:int,use_slurm = False):
        return tf.data.Dataset.from_generator(CropGen(df,image_size,use_slurm).get_next_sample,
                                         output_types={'image':tf.float32,
                                                       'ghi':tf.float32,
                                                       'csky':tf.float32},
                                         output_shapes ={'image':tf.TensorShape([5,image_size,image_size]),
                                                         'ghi':tf.TensorShape([]),
                                                'csky':tf.TensorShape([])}).prefetch(tf.data.experimental.AUTOTUNE)

    
    
class CropGen():
    
    def __init__(self,df:pd.DataFrame,image_size:int,use_slurm=True):
        self.data_path = os.environ["SLURM_TMPDIR"] if use_slurm else '/project/cq-training-1/project1/teams/team12/'
        self.data_path = Path(self.data_path)/f'crops-{image_size}'
        self.df = df
        self.stations = ['BND','TBL','DRA','FPK','GWN', 'PSU','SXF']
        
    def get_next_sample(self):
        f = None
        for index,row in self.df.iterrows():
            if f != index.date():
                f = str(index.date()) 
                arr = np.load(self.data_path/f'{f}.npy',allow_pickle=False,fix_imports=False)
            stamp = arr[row.new_offset]
            for i,station in enumerate(self.stations):
                ghi = row[station+"_GHI"]
                csky = row[station+"_CLEARSKY_GHI"]
                yield({'image':stamp[i],'ghi':ghi,'csky':csky})
                

def benchmark(dataset,epochs=3):
    start_time = time.perf_counter()
    for j in tqdm(range(epochs)):
        for sample in tqdm(dataset, desc=f'Epoch: {j}'):
            time.sleep(0.01) # Simulating a training step
            
    total_time = time.perf_counter() - start_time
    print("Execution time:", total_time)

    
    
def main(subset_size:int=300,
         epochs:int=3,
         do_cache:bool=True,
         copy_data:bool=False): 
    
    
    print('Reading dataframe...')
    metadata = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    
    # Just call preprocessing
    #df = preprocessing.preprocess(df)
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    new_offset= np.load(Path('/project/cq-training-1/project1/teams/team12/crops-20/new_index.npy'))
    metadata = metadata.assign( new_offset = new_offset)
    
    
    
    
    
    # subset
    #metadata = metadata.iloc[:subset_size]
        
#     # Copy files to compute node
#     path = utils.copy_files(data_path='/project/cq-training-1/project1/data/', hdf5_folder='hdf5v7_8bit') if copy_data else '/project/cq-training-1/project1/data/hdf5v7_8bit'
    
    
    # Now try with and without caching, tremendous improvement
    if do_cache:
        benchmark(CropDataset(metadata,image_size=20, use_slurm=False).cache(), epochs=epochs)
    else:
        benchmark(CropDataset(metadata,image_size=20, use_slurm=False), epochs=epochs)
        

    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
    fire.Fire(main)