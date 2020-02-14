import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import math
import os
import shutil
import tarfile
#import utils
import pickle
import time
import tensorflow as tf
from datetime import timedelta
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())


class CropDataset(tf.data.Dataset):
    
    def __new__(cls,df: pd.DataFrame, image_size: int, data_dir: str, delta: list =[15,45,75,105,135], 
                interpolate:bool = True, faire:int =2, fix_shape:bool= True):
       
        """ delta is a list of T-delta timestamps you need. (No need for T0)  
            For No Interpolation -> set interpolate = False 
            For Interpolation w fix shape -> set interpolate = True and fix_shape = True
            For just Interpolation but flexible shape -> set interpolate = True and fix_shape = False
            Fix shape = True: adds zeros if the index is missing, otehwise just skip it.
            'faire' controls the number of timestamps to iterate for interpolation before giving up! 
        """
        
        return tf.data.Dataset.from_generator(CropGen(df, image_size, data_dir).get_next_sample,
                                         # args = delta,interpolate,faire,fix_shape
                                         args = (delta, interpolate,faire,fix_shape),
                                         output_types={ 'images':tf.float32,
                                                        'station':tf.int32,
                                                        'ghi':tf.float32,
                                                        'csky':tf.float32,
                                                        'timestamp':tf.int32},
                                         output_shapes ={'images':tf.TensorShape([None,5,image_size,image_size]),
                                                         'station':tf.TensorShape([7]),
                                                         'ghi':tf.TensorShape([4]),
                                                         'csky':tf.TensorShape([None]),
                                                         'timestamp':tf.TensorShape([None,71]),
                                                        }).prefetch(tf.data.experimental.AUTOTUNE)



class CropGen():
    
    #  TODO flexible crop sizes using 50x50 image as the base + 
    
    def __init__(self,df: pd.DataFrame, image_size: int, data_dir: str):
        self.df = df
        self.image_size = image_size
        self.data_path = Path(data_dir)/f'crops-{image_size}'
        self.stations = ['BND','TBL','DRA','FPK','GWN', 'PSU','SXF']
        
    def enc_timestamp(self, index): # encode timestamps
        # 31 (1-31) - Day, 12 (1-12) - Month, 24 (0-23) - Hours, 4 (0,15,30,45) - Minutes
        enc = np.zeros(71,dtype=np.int)
        enc[index.day-1] = 1
        enc[30+index.month] = 1
        enc[43+index.hour] = 1
        enc[67+[0,15,30,45].index(index.minute)] = 1
        return enc
    
    def nowcast(self, index): 
        """ get T,T+1,T+3,T+6 ghi and interpolate if missing"""
        col_ghi = [s+'_GHI' for s in self.stations]
        ghis = [self.df.loc[index,col_ghi]] # T_0
        for i in [1,3,6]: #TODO replace negative values with zeros
            if index + timedelta(hours=i) in self.df.index:
                new_index = index + timedelta(hours=i)
                ghis.append(self.df.loc[new_index,col_ghi])
            elif index + timedelta(hours=i-1,minutes=45) in self.df.index and index + timedelta(hours=i,minutes=15) in self.df.index:
                g = (self.df.loc[index + timedelta(hours=i-1,minutes=45),col_ghi] + self.df.loc[index + timedelta(hours=i,minutes=15),col_ghi])/2
                ghis.append(g)
            elif index + timedelta(hours=i,minutes=15) in self.df.index:
                new_index = index + timedelta(hours=i,minutes=15)
                ghis.append(self.df.loc[new_index,col_ghi])
            elif index + timedelta(hours=i-1,minutes=45) in self.df.index:
                new_index = index + timedelta(hours=i-1,minutes=45)
                ghis.append(self.df.loc[new_index,col_ghi])
            elif index + timedelta(hours=i,minutes=30) in self.df.index:
                new_index = index + timedelta(hours=i,minutes=30)
                ghis.append(self.df.loc[new_index,col_ghi])
            elif index + timedelta(hours=i-1,minutes=30) in self.df.index:
                new_index = index + timedelta(hours=i-1,minutes=30)
                ghis.append(self.df.loc[new_index,col_ghi])
            else: # TODO improve
                ghis.append([0]*7)
        return np.array(ghis)
    
    def interpolate(self,index,faire):
        """Tries interpolating 'faire' number of times before giving up"""
        tmp_index = index
        for i in range(faire): 
            tmp_index = tmp_index-timedelta(minutes=15) # go 15 mins back
            if tmp_index in self.df.index:
               # print("Put in: ",tmp_index)
                full_day = np.load(self.data_path/f'{str(tmp_index.date())}.npy',allow_pickle=False,fix_imports=False) # open file
                img = full_day[self.df.at[tmp_index,'new_offset']]
                c = self.df.loc[tmp_index,self.col_csky].tolist()
                t = self.enc_timestamp(tmp_index)
                o = 15*(i+1)
                return o,img,c,t
        return 0,None,None,None
        
           
    def fs(self, index, delta:list, interpolate:bool, faire:int, fix_shape:bool):
        """ Fetches T-delta samples 
            For No Interpolation -> set interpolate = False 
            For Interpolation w fix shape -> set interpolate = True and fix_shape = True
            For just Interpolation but flexible shape -> set interpolate = True and fix_shape = False
            Fix shape = True: adds zeros if the index is missing, otehwise just skip it.
            'faire' controls the number of timestamps to iterate for interpolation before giving up! 
        """
        delta = np.insert(delta,0,0)
        tmp_index,f,offset =index,None,0
        seq_images,cskys,timestamps = [],[],[]
        
        for d in delta:
            #print(type(int(d)))
            tmp_index = index - timedelta(minutes=int(d)+offset)
            #print("Current: ", tmp_index)
            if f != tmp_index.date(): # open file
                f = str(tmp_index.date()) 
                full_day = np.load(self.data_path/f'{f}.npy',allow_pickle=False,fix_imports=False)
                
            if tmp_index in self.df.index: # record stuff
                #print("Put in: ",tmp_index)
                seq_images.append(full_day[self.df.at[tmp_index,'new_offset']])
                cskys.append(self.df.loc[tmp_index,self.col_csky].tolist())
                timestamps.append(self.enc_timestamp(tmp_index))
            
            else: 
                if interpolate: 
                    o,img,c,t = self.interpolate(tmp_index,faire)
                    if o != 0 : # Interpolation possible     #TODO Fix missing csky and ghi values in df
                        offset += o
                        seq_images.append(img)
                        cskys.append(c)
                        timestamps.append(t)
                    else: # only if fix shape required
                        if fix_shape:
                            seq_images.append(np.zeros((7,5,self.image_size,self.image_size)))
                            cskys.append([0]*7)
                            timestamps.append(self.enc_timestamp(tmp_index))
                else: # No interpolation + fix shape
                    seq_images.append(np.zeros((7,5,self.image_size,self.image_size)))
                    cskys.append([0]*7)
                    timestamps.append(self.enc_timestamp(tmp_index))
                    
        return np.array(seq_images),np.array(cskys).reshape(-1,7), np.array(timestamps)
        
    def get_next_sample(self,delta:list, interpolate:bool, faire:int, fix_shape:bool):
        """delta is a list of T-delta timestamps you need. Adds T0.
            Look up the discription of  function "fs" """
        
        self.col_csky = [s+'_CLEARSKY_GHI' for s in self.stations] # just to pick out clearsky values
        for index,row in self.df.iterrows():
            
            seq_images,cskys,timestamps = self.fs(index,delta,interpolate,faire,fix_shape)

            # seq_images -> size: nums*7*size*size
            # cskys -> nums * 7
            # timestamps -> nums*71
            ghis = self.nowcast(index) # size:4x7

            for i,station in enumerate(self.stations):
                ghi = ghis[:,i]
                csky = cskys[:,i]
                images = seq_images[:,i]
                yield({ 'images':images,
                        'station':np.eye(7,dtype=np.int)[i],
                        'ghi':ghi,
                        'csky':csky,
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
         do_cache:bool=True,
         use_slurm:bool=True): 
    
    print('Reading dataframe...')
    metadata = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    new_offset= np.load(Path('/project/cq-training-1/project1/teams/team12/crops-20/new_index.npy'))
    metadata = metadata.assign( new_offset = new_offset)
    
 
     
    dest = os.environ['SLURM_TMPDIR'] if use_slurm else '/project/cq-training-1/project1/teams/team12/'  
    if use_slurm: metadata = crops.get_crops(df = metadata,stations = None,image_size=20,dest = dest) # copy to slurm
    
    # subset
    metadata = metadata.iloc[:subset_size]
    
    # Some arguments 
    delta = [15,45,75,105,135]
    interpolate = True
    fix_shape = True
    faire =2
    # Now try with and without caching, tremendous improvement
    if do_cache:
        benchmark(CropDataset(metadata,image_size=20,data_dir = dest,delta = delta,interpolate=interpolate,
                              faire=faire,fix_shape=fix_shape).cache(), epochs=epochs)
    else:
        benchmark(CropDataset(metadata,image_size=20,data_dir = dest,delta=delta,interpolate=interpolate,
                              faire=faire,fix_shape=fix_shape), epochs=epochs)
 

                  
if __name__ == "__main__":
                  
    import fire
    import sys
    sys.path.append('../utils/')
    import crops
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
    fire.Fire(main)