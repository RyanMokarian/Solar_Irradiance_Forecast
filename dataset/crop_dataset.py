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
import time
import tensorflow as tf
from datetime import timedelta
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())

class CropDataset(tf.data.Dataset):
    
    def __new__(cls,df: pd.DataFrame, image_size: int, data_dir: str, num_seq: int = 5, flexible_seq: bool = False):
        if flexible_seq == True:
            raise Exception(f'Flexible sequences not completely implemented yet.')
        
        return tf.data.Dataset.from_generator(CropGen(df, image_size, data_dir, flexible_seq).get_next_sample,
                                         args = [num_seq],
                                         output_types={ 'images':tf.float32,
                                                        'station':tf.int32,
                                                        'ghi':tf.float32,
                                                        'csky':tf.float32,
                                                        'csky_ghi':tf.float32,
                                                        'timestamp':tf.int32},
                                         output_shapes ={'images':tf.TensorShape([num_seq,image_size,image_size,5]),
                                                         'station':tf.TensorShape([7]),
                                                         'ghi':tf.TensorShape([4]),
                                                         'csky':tf.TensorShape([None]),
                                                         'csky_ghi':tf.TensorShape([4]),
                                                         'timestamp':tf.TensorShape([None,71]),
                                                        }).prefetch(tf.data.experimental.AUTOTUNE).cache()

class CropGen():
    
    #  TODO flexible crop sizes using 50x50 image as the base + 
    
    def __init__(self,df: pd.DataFrame, image_size: int, data_dir: str, flexible_seq: bool = False):
        self.df = df
        self.image_size = image_size
        self.data_path = Path(data_dir)/f'crops-{image_size}'
        self.flexible_seq = flexible_seq
        self.stations = ['BND','TBL','DRA','FPK','GWN', 'PSU','SXF']
        self.col_csky = [s+'_CLEARSKY_GHI' for s in self.stations] # just to pick out clearsky values
        self.col_ghi = [s+'_GHI' for s in self.stations]
        
    def enc_timestamp(self, index): # encode timestamps
        # 31 (1-31) - Day, 12 (1-12) - Month, 24 (0-23) - Hours, 4 (0,15,30,45) - Minutes
        enc = np.zeros(71,dtype=np.int)
        enc[index.day-1] = 1
        enc[30+index.month] = 1
        enc[43+index.hour] = 1
        enc[67+[0,15,30,45].index(index.minute)] = 1
        return enc
    
    # def nowcast(self, index): 
    #     """ get T,T+1,T+3,T+6 ghi and interpolate if missing"""
    #     csky_ghis = [self.df.loc[index,self.col_csky]] # T_0
    #     df = self.df.replace('nan',np.NaN).dropna()
    #     ghis = [df.loc[index,col_ghi]] # T_0
    #     for i in [1,3,6]: #TODO replace negative values with zeros
    #         if index + timedelta(hours=i) in df.index:
    #             new_index = index + timedelta(hours=i)
    #             ghis.append(df.loc[new_index,col_ghi])
    #             csky_ghis.append(df.loc[new_index,self.col_csky])
    #         elif index + timedelta(hours=i-1,minutes=45) in df.index and index + timedelta(hours=i,minutes=15) in df.index:
    #             g = (df.loc[index + timedelta(hours=i-1,minutes=45),col_ghi] + df.loc[index + timedelta(hours=i,minutes=15),col_ghi])/2
    #             ghis.append(g)
    #             csky_ghis.append((df.loc[index + timedelta(hours=i-1,minutes=45),self.col_csky]+ df.loc[index + timedelta(hours=i,minutes=15),self.col_csky])/2)
    #         elif index + timedelta(hours=i,minutes=15) in df.index:
    #             new_index = index + timedelta(hours=i,minutes=15)
    #             ghis.append(df.loc[new_index,col_ghi])
    #             csky_ghis.append(df.loc[new_index,self.col_csky])
    #         elif index + timedelta(hours=i-1,minutes=45) in df.index:
    #             new_index = index + timedelta(hours=i-1,minutes=45)
    #             ghis.append(df.loc[new_index,col_ghi])
    #             csky_ghis.append(df.loc[new_index,self.col_csky])
    #         elif index + timedelta(hours=i,minutes=30) in df.index:
    #             new_index = index + timedelta(hours=i,minutes=30)
    #             ghis.append(df.loc[new_index,col_ghi])
    #             csky_ghis.append(df.loc[new_index,self.col_csky])
    #         elif index + timedelta(hours=i-1,minutes=30) in df.index:
    #             new_index = index + timedelta(hours=i-1,minutes=30)
    #             ghis.append(df.loc[new_index,col_ghi])
    #             csky_ghis.append(df.loc[new_index,self.col_csky])
    #         else: # TODO improve
    #             ghis.append([0]*7)
    #             csky_ghis.append([0]*7)
    #     return np.array(ghis), np.array(csky_ghis)

    def nowcast(self, index): 
        """ 
        Gets (T, T+1, T+3, T+6) GHIs and interpolate values if missing.
        """
        def time_exist(timestamp, cols):
            return timestamp in self.df.index and not self.df.loc[timestamp, cols].isna().any()
        ghis, csky_ghis = [], []
        timestamps = np.array([0, 1, 3, 6]) * timedelta(hours=1) + index # T, T+1, T+3, T+6
        for l, cols in ((ghis, self.col_ghi), (csky_ghis, self.col_csky)):
            for timestamp in timestamps:
                if time_exist(timestamp, cols):
                    l.append(self.df.loc[timestamp,cols])
                    continue
                found = False
                for step in np.array([0, 1, 2, 3]) * timedelta(minutes=15):
                    if time_exist(timestamp+step, cols) and time_exist(timestamp-step, cols):
                        l.append((self.df.loc[timestamp+step,cols]+self.df.loc[timestamp-step,cols])/2)
                        found = True
                        break
                if not found:
                    l.append([0]*7)  
        return np.array(ghis), np.array(csky_ghis)
    
    def fetch_flexible_sequence(self,index,num,limit=10):
        """
        Get 'num' samples sequence data, with a variable timestep depending on data availability.

        Returns:
            (np.array, np.array, np.array) -- Images, csky ghis, timesteps
        """
        i,trials = 0,0
        tmp_index,f = index, None
        seq_images,cskys,timestamps = [],[],[]
        while i<num and trials <limit: # trials sets a hard limit on while excecution (in case no sample found or missing)
            # open file
            if f != tmp_index.date():
                f = str(tmp_index.date()) 
                full_day = np.load(self.data_path/f'{f}.npy',allow_pickle=False,fix_imports=False)
            # record stuff
            if tmp_index in self.df.index:     
                seq_images.append(full_day[self.df.at[tmp_index,"new_offset"]])
                cskys.append(self.df.loc[tmp_index,self.col_csky])
                timestamps.append(self.enc_timestamp(tmp_index))
                i += 1
            # update index - 15 minutes
            tmp_index = tmp_index - timedelta(minutes=15)
            trials +=1

        return np.array(seq_images),np.array(cskys),np.array(timestamps)

    def fetch_hard_sequence(self,index,num):
        """
        Get 'num' samples sequence data, with image interpolation (Not implemented yet) depending on data availability.

        Returns:
            (np.array, np.array, np.array) -- Images, csky ghis, timestamps
        """
        tmp_index, f = index, None
        seq_images, cskys, timestamps = np.zeros((num, 7, 5, self.image_size, self.image_size)), np.zeros((num, 7)), []
        for i in range(len(seq_images)):
            if f != tmp_index.date():
                f = str(tmp_index.date()) 
                full_day = np.load(self.data_path/f'{f}.npy',allow_pickle=False,fix_imports=False)
            if tmp_index in self.df.index: 
                seq_images[i] = full_day[self.df.at[tmp_index,"new_offset"]]
                cskys[i] = self.df.loc[tmp_index,self.col_csky] # TODO : We dropped columns that we shouldn't have maybe ? 
                tmp_index = tmp_index - timedelta(minutes=15)
            timestamps.append(self.enc_timestamp(tmp_index))
        
        seq_images = np.moveaxis(seq_images, 2, -1) # Move channels last

        return seq_images, cskys, np.array(timestamps)
            
    def get_next_sample(self,num=5):
        
        for index,row in self.df.iterrows():
            if self.flexible_seq:
                seq_images, cskys, timestamps = self.fetch_sequence(index, num)
            else:
                seq_images, cskys, timestamps = self.fetch_hard_sequence(index, num)
            # seq_images -> size: nums*7*size*size
            # cskys -> nums * 7
            # timestamps -> nums*71

            if self.df.loc[index,self.col_ghi].isna().any(): # Continue if T0 is nan
                continue

            ghis, future_cskys = self.nowcast(index) # size:4x7

            for i,station in enumerate(self.stations):
                ghi = ghis[:,i]
                csky = cskys[:,i]
                future_csky = future_cskys[:,i]
                images = seq_images[:,i]
                yield({ 'images':images,
                        'station':np.eye(7,dtype=np.int)[i],
                        'ghi':ghi,
                        'csky':csky,
                        'csky_ghi':future_csky,
                        'timestamp':timestamps})

def benchmark(dataset,epochs=3):
    start_time = time.perf_counter()
    for j in tqdm(range(epochs)):
        for sample in tqdm(dataset, desc=f'Epoch: {j}'):
            time.sleep(0.01) # Simulating a training step
            
    total_time = time.perf_counter() - start_time
    print("Execution time:", total_time)

def main(subset_size:int=300,
         epochs:int=3,
         num_seq:int=6,
         do_cache:bool=True,
         use_slurm:bool=True): 
    
    print('Reading dataframe...')
    metadata = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    new_offset= np.load(Path('/project/cq-training-1/project1/teams/team12/crops-20/new_index.npy'))
    metadata = metadata.assign( new_offset = new_offset)
    
    # subset
    metadata = metadata.iloc[:subset_size]
        
    if use_slurm: crops.get_crops(metadata) # copy to slurm
    
    # Now try with and without caching, tremendous improvement
    if do_cache:
        benchmark(CropDataset(metadata,image_size=20,num_seq=num_seq,use_slurm=use_slurm).cache(), epochs=epochs)
    else:
        benchmark(CropDataset(metadata,image_size=20,num_seq=num_seq,use_slurm=use_slurm), epochs=epochs)
    
if __name__ == "__main__":
    import fire
    import crops
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
    fire.Fire(main)