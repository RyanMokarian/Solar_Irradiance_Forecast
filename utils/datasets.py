
# -------- Code to benchmark dataset efficency --------- #
from __future__ import absolute_import, division, print_function, unicode_literals
if __name__ == "__main__":
    import utils
    import pickle
    from tqdm import tqdm
    import preprocessing
    import time
    import fire
else:
# ------------------------------------------------------ #
    from utils import utils
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py


class SolarIrradianceDataset(tf.data.Dataset):
    """
    Dataset that loads satellite imagery from HDF5 files.
    """    
    def __new__(cls, df : pd.DataFrame, image_size : int, data_path: str = None):
        return tf.data.Dataset.from_generator(DataGenerator(df, image_size, data_path).get_next_example,
                                            output_types={'hdf5_8bit_path': tf.string,
                                                        'hdf5_8bit_offset': tf.int32,
                                                        'station_name': tf.string,
                                                        'station_lat': tf.float32,
                                                        'station_long': tf.float32,
                                                        'images': tf.float32,
                                                        'csky_ghi': tf.float32,
                                                        'ghi': tf.float32},
                                             output_shapes={'hdf5_8bit_path': tf.TensorShape([]),
                                                        'hdf5_8bit_offset': tf.TensorShape([]),
                                                        'station_name': tf.TensorShape([]),
                                                        'station_lat': tf.TensorShape([]),
                                                        'station_long': tf.TensorShape([]),
                                                        'images': tf.TensorShape([image_size, image_size, 5]),
                                                        'csky_ghi': tf.TensorShape([]),
                                                        'ghi': tf.TensorShape([])}).prefetch(tf.data.experimental.AUTOTUNE)

class DataGenerator(object):
    """
    Generator that yields one training example at a time.
    """
    def __init__(self, df: pd.DataFrame, image_size: int, data_path: str):
        self.df = df
        self.image_size = image_size
        self.data_path = data_path
        self.images_mean = {'ch1': 0.3154958181516178, 'ch2': 274.18017705595855, 
                            'ch3': 230.33078484969695, 'ch4': 264.2658065947056, 
                            'ch6': 245.69855600075982}
        self.images_std = {'ch1': 0.30688239459685107, 'ch2': 60.6773046722939, 
                           'ch3': 50.38130188316295, 'ch4': 59.25418221268753, 
                           'ch6': 54.37427451580163}
        self.stations = {'BND':(40.05192,-88.37309), 'TBL':(40.12498,-105.2368),
                         'DRA':(36.62373,-116.01947), 'FPK':(48.30783,-105.1017),
                         'GWN':(34.2547,-89.8729), 'PSU':(40.72012,-77.93085), 
                         'SXF':(43.73403,-96.62328)}

    def get_next_example(self):
        # TODO : shuffle dataframe while keeping examples from the same file together (to improve performance)
        
        # Iterate over all rows of the dataframe
        open_path = None
        for index, row in self.df.iterrows():
            hdf5_8bit_path = row['hdf5_8bit_path'] if self.data_path is None else os.path.join(self.data_path, row['hdf5_8bit_path'].split('/')[-1])
            hdf5_8bit_offset = row['hdf5_8bit_offset']
            
            # Open hdf5 file if it is not already opened
            if open_path != hdf5_8bit_path:
                if open_path != None:
                    h5_data.close()
                h5_data = h5py.File(hdf5_8bit_path, "r")
                open_path = hdf5_8bit_path
                
            # Get latitude & longitude stored in the file
            lats, lons = utils.fetch_hdf5_sample("lat", h5_data, hdf5_8bit_offset), utils.fetch_hdf5_sample("lon", h5_data, hdf5_8bit_offset)
            if lats is None or lons is None:
                print(f'WARNING : latlong of date {index} are unavailable.')
                continue
                    
            # Get data from the 5 channels
            images = []
            for channel in ('ch1', 'ch2', 'ch3', 'ch4', 'ch6'):
                img = utils.fetch_hdf5_sample(channel, h5_data, hdf5_8bit_offset)
                if type(img) is np.ndarray:
                    images.append(img)
            if len(images) < 5:
                print(f'WARNING : {5-len(images)} channels are not available at date {index}')
                continue
                
            # Yield relevant data for each station
            for station_name, station_coords in self.stations.items():
                
                # Skip station if it is night
                if row[station_name + "_DAYTIME"] == 0:
                    continue
                
                # Get pixel coords
                pixel_coords = (np.argmin(np.abs(lats - station_coords[0])), np.argmin(np.abs(lons - station_coords[1])))
                ghi = row[station_name + "_GHI"]
                csky_ghi = row[station_name + "_CLEARSKY_GHI"]
                    
                # Crop the images with the station centered
                pixels = self.image_size//2
                adjustement = self.image_size % 2 # Adjustement if image_size is odd
                cropped_images = []
                for img, mean, std in zip(images, self.images_mean.values(), self.images_std.values()):
                    # TODO : Check if the slice is out of bounds
                    img = (img - mean)/std # Normalize image
                    cropped_images.append(img[pixel_coords[0]-pixels:pixel_coords[0]+pixels+adjustement,
                                          pixel_coords[1]-pixels:pixel_coords[1]+pixels+adjustement])
                cropped_images = tf.convert_to_tensor(np.moveaxis(np.array(cropped_images), 0, -1))
                    
                yield ({'hdf5_8bit_path': hdf5_8bit_path, 
                        'hdf5_8bit_offset': hdf5_8bit_offset,
                        'station_name': station_name,
                        'station_lat': pixel_coords[0],
                        'station_long': pixel_coords[1],
                        'images': cropped_images,
                        'csky_ghi': csky_ghi,
                        'ghi': ghi})


# ----- Code to benchmark dataset efficency ----- #

# change examples to epochs to check if caching helps, need to iterate entirety
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache

def benchmark(dataset,epochs=3):
    start_time = time.perf_counter()
    for j in tqdm(range(epochs)):
        for sample in tqdm(dataset, desc=f'Epoch: {j}'):
            time.sleep(0.01) # Simulating a training step
            
    total_time = time.perf_counter() - start_time
    print("Execution time:", total_time)
    #print("Execution time per example:", total_time/(num_examples*epochs))
    #print("Number of example per second:", 1 / (total_time/(num_examples*epochs)))
    
def main(subset_size:int=300,
         epochs:int=3,
         do_cache:bool=True,
         copy_data:bool=False): 
    
    
    print('Reading dataframe...')
    df = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    
    # Just call preprocessing
    df = preprocessing.preprocess(df)
    
    # subset
    df = df.iloc[:subset_size]
        
    # Copy files to compute node
    path = utils.copy_files(data_path='/project/cq-training-1/project1/data/', hdf5_folder='hdf5v7_8bit') if copy_data else '/project/cq-training-1/project1/data/hdf5v7_8bit'
    
    
    # Now try with and without caching, tremendous improvement
    if do_cache:
        benchmark(SolarIrradianceDataset(df,image_size=20, data_path=path).cache(), epochs=epochs)
    else:
        benchmark(SolarIrradianceDataset(df,image_size=20, data_path=path), epochs=epochs)
        

    
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
    fire.Fire(main)

