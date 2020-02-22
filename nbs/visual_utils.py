import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle
from pathlib import Path
from tqdm import tqdm
import math
import utils
import tensorflow as tf

def lr_find(model,train_dataset):
    beta,mv = 0.95,0
    tloss,best =[],100
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.MSE
    lrs = np.logspace(-7,0,150)
    for i,(lr,data) in enumerate(tqdm(zip(lrs,train_dataset),total=150)):
        optimizer.lr = lr
        samples,labels = data['images'],data['ghi']
        samples = tf.squeeze(samples,axis=1)
        labels = labels[:,0]
        labels = (labels - 188.5)/285
        with tf.GradientTape() as tape:
            preds = tf.keras.backend.flatten(model(samples,training=True))
            loss = loss_func(labels,preds)
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        mv = (beta*mv + (1-beta)*loss.numpy())
        av = mv / (1-beta**(i+1))
        #print(av.shape)
        best = min(best,av)
        if  av > 4*best and i > 50:
            print("broken at: ", i)
            break
        tloss.append(av)
    plt.plot(lrs[:len(tloss)],tloss)
    plt.xscale('log')
    plt.ylabel("Loss")
    plt.xlabel("Learning Rate")
    plt.show()


def show_image(ax,img,title=None):
    """Takes in plt.axs, an image and title and plots the image"""
    ax.imshow(img,cmap='bone')
    ax.axis('off')
    ax.set_title(title)
    
    
def plot_images(df:pd.DataFrame,n:int,idxs:list,channels:str):
    """Sets up plt.axes and figures to plot images acoording to the channel required """
    if len(channels) == 1:
        _,axs = plt.subplots(math.ceil(n/3),3,figsize=(15,n))
        axs = axs.flatten()
        plt.suptitle(f"Channel:{channels}",y=0.9)
        for i,idx in enumerate(idxs):
            img = get_channel_data(df.loc[df.index[idx]],channels)
            show_image(axs[i],img,df.index[idx])

    else:
        l_ch = len(channels.split(','))
        _,axs = plt.subplots(n,l_ch,figsize=((l_ch)*4,n*2.5))
        plt.suptitle(f"Channels:{channels}",y=0.9)
        for i,idx in enumerate(idxs):
            imgs = get_channel_data(df.loc[df.index[idx]],channels)
            for j,img in enumerate(imgs):
                try:
                    show_image(axs[i,j],img,df.index[idx])
                except:
                    print("Error at:",df.index[idx])

    
def get_channel_data(row:pd.Series,chs='1,2,3,4,6'):
    "Takes a dataframe row and returns data for the corresponding channels"
    f = h5py.File(row.hdf5_8bit_path,'r')
    offset = row.hdf5_8bit_offset
    if len(chs) >1:
        chs = ['ch'+str(c) for c in chs.split(',')]
        return [utils.fetch_hdf5_sample(ch,f,offset) for ch in chs]            
    else:
        return utils.fetch_hdf5_sample('ch'+str(chs),f,offset)

    
def req_indices(df:pd.DataFrame,n:int=5, start_time:str = 'random', sequence:bool = False):
    """Flexible function to get random indices and timestamp in sequence returns indices of the dataframe"""
    if start_time != 'random':
        try:
            start = np.flatnonzero(df.index == start_time)[0]
            idxs = list(range(start,start+n))
        except:
            print("Incorrect or Missing TimeStamp")
            return
    elif start_time == 'random' and sequence == True:
        start = np.random.choice(len(df)-n)
        idxs = list(range(start,start+n))
    else:
        idxs = np.random.choice(len(df),n,replace=False)

    return idxs


def visualize(df:pd.DataFrame,n:int=5, start_time:str = 'random', sequence:bool = False, channels:str='1,2,3,4,6'):
    """ Dummy function to get indices and visualize the channels"""
    idxs = req_indices(df,n,start_time,sequence)
    if idxs is None:
        return
    plot_images(df,n,idxs,channels)

    
def plot_pixel_range(df:pd.DataFrame,s:int=1000,n:int=9,start:int=0,stop:int=15):
    """"Samples 's' points, plots 'n' random images that are between start and stop unique pixels."""
    req_idx= []
    idxs = req_indices(df,s)
    for idx in idxs:
        img = get_channel_data(df.loc[df.index[idx]],'1')
        if len(np.unique(img))>start and  len(np.unique(img)) <=stop:
            if len(req_idx)==n:
                break
            req_idx.append(idx)

    plot_images(df,n,req_idx,channels='1')


def get_hdf5_data(row:pd.Series,key:'str'):
    "Fetches the value inside 'key' of an hdf5 file "
    f = h5py.File(row.hdf5_8bit_path,'r')
    offset = row.hdf5_8bit_offset
    return utils.fetch_hdf5_sample(key,f,offset)



def image_pixels(row:pd.Series,stations:dict):
    """Returns the pixel location of each station in an image using latitude and logitude"""
    lat = get_hdf5_data(row,'lat')
    lon = get_hdf5_data(row,'lon')
    station_coords ={}
    for key in stations:
        station_coords[key]  = (np.argmin(np.abs(lat - stations[key][0])), np.argmin(np.abs(lon - stations[key][1])))
    return station_coords

def image_pixels(row:pd.Series,stations:dict):
    """Returns the pixel location of each station in an image using latitude and logitude"""
    lat = get_hdf5_data(row,'lat')
    lon = get_hdf5_data(row,'lon')
    station_coords ={}
    for key in stations:
        station_coords[key]  = (np.argmin(np.abs(lat - stations[key][0])), np.argmin(np.abs(lon - stations[key][1])))
    return station_coords


def plot_stations(row:pd.Series,stations:dict):
    """Plot the stations on the image"""
    imgs = get_channel_data(row)
    pixs = image_pixels(row,stations)
    _,axs = plt.subplots(2,3,figsize=(20,5))
    axs = axs.flatten()
    for i,img in enumerate(imgs):
        show_image(axs[i],img,f"Channel:{i+1}")
        for j,key in enumerate(pixs):
            axs[i].add_patch(Circle((pixs[key][1],pixs[key][0]),10,color=utils.get_label_color_mapping(j+1)/255))
            # Take care of  plotting  x,y  --> j,i  array index  

def get_crops(row:pd.Series,stations:dict,size:int=30):
    """Get Crops of 'size' pixels for each station and plots the crops with stations"""
    img = get_channel_data(row,chs='1')
    pixs = image_pixels(row,stations)
    _,axs = plt.subplots(2,4,figsize=(8,4))
    axs = axs.flatten()
    for i,key in enumerate(pixs):
        n_img = img[pixs[key][0]-size//2:pixs[key][0]+size//2,pixs[key][1]-size//2:pixs[key][1]+size//2]
        show_image(axs[i],n_img,key)
        axs[i].add_patch(Circle((size//2,size//2),2,color=utils.get_label_color_mapping(i+1)/255))
        
    axs[7].axis('off')