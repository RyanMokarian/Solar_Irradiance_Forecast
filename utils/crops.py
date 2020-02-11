import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import math
import os
import shutil
import tarfile
import utils

from pathlib import Path
Path.ls = lambda x:list(x.iterdir())



stations = {'BND':(40.05192,-88.37309), 'TBL':(40.12498,-105.2368),
                         'DRA':(36.62373,-116.01947), 'FPK':(48.30783,-105.1017),
                         'GWN':(34.2547,-89.8729), 'PSU':(40.72012,-77.93085), 
                         'SXF':(43.73403,-96.62328)}

def create_crops(df,stations = stations ,size = 20,dest='.'):
    """Function to create crops and save as .npy"""
    # store in new_dir 'crops-size'
    dest = Path(dest+'/crops-'+str(size))
    dest.mkdir()
    pixels= math.ceil(size//2) # always even
    #missing data, data for an entire day, new indexing
    missing_data,curr_day,new_idx =[],[],[]  
    f_name = str(df.index[0].date()) # set file_name as first date
    for index,row in tqdm(df.iterrows(),total = len(df)):
        if f_name != str(index.date()): # if date's don't match save
            if curr_day: # save only if current data is not empty
                new_idx = new_idx+list(range(len(curr_day))) # append new index mapping
                np.save(dest/f_name,np.array(curr_day),allow_pickle=False,fix_imports=False) 
            curr_day = [] # reset current day data
            f_name = str(index.date()) # new-file-name
        offset = row.hdf5_8bit_offset
        f = h5py.File(row.hdf5_8bit_path,'r')
        lats,lons = utils.fetch_hdf5_sample('lat',f,offset), utils.fetch_hdf5_sample('lon',f,offset)
        images = [] # save channel info in list
        for key in ('ch1','ch2','ch3','ch4','ch6'):
            img = utils.fetch_hdf5_sample(key,f,offset)
            if isinstance(img,np.ndarray): images.append(img)
        if len(images) < 5:  #  if some channel missing
            missing_data.append(index)
            new_idx.append(np.NaN)  # adjust index
            continue
        images = np.array(images)  # Channels x img-size x img-size
        crops = []  # data for all  7 stations -  stations x Channels x crop-size x crop-size
        for name,coords in stations.items(): # create the crops
            pixel_loc = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
            crops.append(images[:,pixel_loc[0]-pixels:pixel_loc[0]+pixels,pixel_loc[1]-pixels:pixel_loc[1]+pixels])
        curr_day.append(np.array(crops)) # observations/day x stations x Channels x crop-size x crop-size
    
    # Save the last file
    if curr_day:
        new_idx = new_idx+list(range(len(curr_day))) # append new index mapping
        np.save(dest/f_name,np.array(curr_day),allow_pickle=False,fix_imports=False) 
    np.save(dest/'new_index',np.array(new_idx),allow_pickle=False,fix_imports=False) # save the new-index
    if missing_data: print(f"These rows are missing: {missing_data}")
    return missing_data




def get_crops(df:pd.DataFrame,stations:dict=stations,size:int=20,use_slurm = True,dest=None):
    
    """ Checks for data at /project/cq-training-1/project1/teams/team12/
    creates and tars if missing.
    optionally copies to SLURM  or dest 
    """
    
    dest = os.environ["SLURM_TMPDIR"] if use_slurm else dest
    tmp = Path(str(dest))/f'crops-{size}'
    store = Path('/project/cq-training-1/project1/teams/team12/')
    if tmp.exists(): 
        if len(tmp.ls()) == len((store/f'crops-{size}').ls()):
            print("Data present in destination")
            return
        else: 
            print(f"Deleting: {tmp.name}")
            shutil.rmtree(str(tmp))
            
    f_list = [o.name for o in store.ls()]
    if f'crops-{size}.tar' in f_list:  # If tar exists
        if dest:
            print("Copying Tar")
            shutil.copy(str(store/f'crops-{size}.tar'),dest)
            print("Extracting")
            with tarfile.open(Path(dest)/f'crops-{size}.tar') as tarf:
                tarf.extractall(dest)
        else: 
            print(f"Data present at {store}")
    elif f'crops-{size}' in f_list: # Just copy the folder
        if dest:
            print("Copying Folder")
            utils.copy_files(str(store),f'crops-{size}')
        else: 
            print(f"Data present at {store}")
    else:     # Create the crops,tar,copy,
        print("Creating Crops")
        create_crops(df,stations,size,str(store))
        with tarfile.open(f'crops-{size}.tar','w') as tarf:
            tarf.add(store/f'crops-{size}',f'crops-{size}')
        if dest:
            print("Copying files")
            shutil.copy(str(store/f'crops-{size}.tar'),dest)
            with tarfile.open(Path(dest)/f'crops-{size}.tar') as tarf:
                tarf.extractall(dest)
            

def main(size:int=20,use_slurm=True):
    
    print('Reading dataframe...')
    metadata = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    get_crops(metadata,stations,size,use_slurm)
    
    
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)