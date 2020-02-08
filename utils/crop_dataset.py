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
from datetime import timedelta
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())





class CropDataset(tf.data.Dataset):
    
    def __new__(cls,df:pd.DataFrame,image_size:int,use_slurm = True):
        
        return tf.data.Dataset.from_generator(CropGen(df,image_size,use_slurm).get_next_sample,
                                         output_types={'image':tf.float32,
                                                       'ghi':tf.float32,
                                                       'csky':tf.float32},
                                         output_shapes ={'image':tf.TensorShape([5,image_size,image_size]),
                                                         'ghi':tf.TensorShape([4]),
                                                'csky':tf.TensorShape([4])}).prefetch(tf.data.experimental.AUTOTUNE)

    
    

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
                images = np.load(self.data_path/f'{f}.npy',allow_pickle=False,fix_imports=False)
            ghis,cskys = self.nowcast(index)    
            for i,station in enumerate(self.stations):
                ghi = ghis[:,i]
                csky = cskys[:,i]
                yield({'image':images[row.new_offset,i],'ghi':ghi,'csky':csky})
    
    def nowcast(self,index):
        col_ghi = [s+'_GHI' for s in self.stations]
        col_csky = [s+'_CLEARSKY_GHI' for s in self.stations]
        ghis = [self.df.loc[index,col_ghi]]
        cskys = [self.df.loc[index,col_csky]]
        for i in [1,3,6]: #TODO deal with negatives and nans if present in df
            if index + timedelta(hours=i) in self.df.index:
                new_index = index + timedelta(hours=i)
                ghis.append(self.df.loc[new_index,col_ghi])
                cskys.append(self.df.loc[new_index,col_csky])
            elif index + timedelta(hours=i-1,minutes=45) in self.df.index and index + timedelta(hours=i,minutes=15) in self.df.index:
                g = (self.df.loc[index + timedelta(hours=i-1,minutes=45),col_ghi] + self.df.loc[index + timedelta(hours=i,minutes=15),col_ghi])/2
                ghis.append(g)
                c = (self.df.loc[index + timedelta(hours=i-1,minutes=45),col_csky] + self.df.loc[index + timedelta(hours=i,minutes=15),col_csky])/2
                cskys.append(c)
            elif index + timedelta(hours=i,minutes=15) in self.df.index:
                new_index = index + timedelta(hours=i,minutes=15)
                ghis.append(self.df.loc[new_index,col_ghi])
                cskys.append(self.df.loc[new_index,col_csky])
            elif index + timedelta(hours=i-1,minutes=45) in self.df.index:
                new_index = index + timedelta(hours=i-1,minutes=45)
                ghis.append(self.df.loc[new_index,col_ghi])
                cskys.append(self.df.loc[new_index,col_csky])
            elif index + timedelta(hours=i,minutes=30) in self.df.index:
                new_index = index + timedelta(hours=i,minutes=30)
                ghis.append(self.df.loc[new_index,col_ghi])
                cskys.append(self.df.loc[new_index,col_csky])
            elif index + timedelta(hours=i-1,minutes=30) in self.df.index:
                new_index = index + timedelta(hours=i-1,minutes=30)
                ghis.append(self.df.loc[new_index,col_ghi])
                cskys.append(self.df.loc[new_index,col_csky])
            else:
                ghis.append([0]*7)
                cskys.append([0]*7)
        return np.array(ghis),np.array(cskys)
                
                

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
    
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    new_offset= np.load(Path('/project/cq-training-1/project1/teams/team12/crops-20/new_index.npy'))
    metadata = metadata.assign( new_offset = new_offset)
    
    
    
    
    
    # subset
    metadata = metadata.iloc[:subset_size]
        
#     # Copy files to compute node
#     path = utils.copy_files(data_path='/project/cq-training-1/project1/data/', hdf5_folder='hdf5v7_8bit') if copy_data else '/project/cq-training-1/project1/data/hdf5v7_8bit'
    
    
    # Now try with and without caching, tremendous improvement
    if do_cache:
        benchmark(CropDataset(metadata,image_size=20, use_slurm=True).cache(), epochs=epochs)
    else:
        benchmark(CropDataset(metadata,image_size=20, use_slurm=True), epochs=epochs)
        

    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
    fire.Fire(main)