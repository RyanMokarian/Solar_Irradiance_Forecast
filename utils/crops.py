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

def add_new_offset(df: pd.DataFrame, data_path: str, crop_folder: str):
    """
    Adds new offset to the dataframe to deal with saved npy pre-cropped data.
    """
    new_offset= np.load(os.path.join(data_path, f'{crop_folder}/new_index.npy'))
    df = df.assign(new_offset=new_offset)    
    return df

def create_crops(df, stations, size, dest):
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

def get_crops(df:pd.DataFrame, stations:dict, image_size:int, dest=None):
    
    """ Checks for data at /project/cq-training-1/project1/teams/team12/
    creates and tars if missing. Optionally copies to dest.
    
    Returns:
        pd.DataFrame -- dataframe with new offset indexes column for npy data
    """
    crop_folder = f'crops-{image_size}'
    tmp = Path(str(dest))/crop_folder
    store = Path('/project/cq-training-1/project1/teams/team12/')
    if tmp.exists(): 
        if len(tmp.ls()) == len((store/crop_folder).ls()):
            print("Data present in destination")
            return add_new_offset(df, store, crop_folder)
        else: 
            print(f"Deleting: {tmp.name}")
            shutil.rmtree(str(tmp))
            
    f_list = [o.name for o in store.ls()]
    if f'{crop_folder}.tar' in f_list:  # If tar exists
        if dest:
            print("Copying Tar")
            shutil.copy(str(store/f'{crop_folder}.tar'),dest)
            print("Extracting")
            with tarfile.open(Path(dest)/f'{crop_folder}.tar') as tarf:
                tarf.extractall(dest)
        else: 
            print(f"Data present at {store}")
    elif crop_folder in f_list: # Just copy the folder
        if dest:
            print("Copying Folder")
            utils.copy_files(str(store),crop_folder)
        else: 
            print(f"Data present at {store}")
    else:     # Create the crops,tar,copy,
        print("Creating Crops")
        create_crops(df,stations,size,str(store))
        with tarfile.open(f'{crop_folder}.tar','w') as tarf:
            tarf.add(store/crop_folder,crop_folder)
        if dest:
            print("Copying files")
            shutil.copy(str(store/f'{crop_folder}.tar'),dest)
            with tarfile.open(Path(dest)/f'{crop_folder}.tar') as tarf:
                tarf.extractall(dest)

    df = add_new_offset(df, store, crop_folder)
    return df


def main(size:int=20,stations=None):
    
    print('Reading dataframe...')
    metadata = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    metadata = metadata.replace('nan',np.NaN)
    metadata = metadata[metadata.ncdf_path.notna()]
    get_crops(metadata,stations,size,os.environ["SLURM_TMPDIR"])
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)