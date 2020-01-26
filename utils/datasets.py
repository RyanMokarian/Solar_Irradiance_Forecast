
# -------- Code to benchmark dataset efficency --------- #
from __future__ import absolute_import, division, print_function, unicode_literals
if __name__ == "__main__":
    import utils
else:
# ------------------------------------------------------ #
    from utils import utils
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py

class SolarIrradianceDataset(tf.data.Dataset):
    """
    Dataset that loads satellite imagery from HDF5 files.
    """    
    def __new__(cls, df : pd.DataFrame, image_size : int):
        return tf.data.Dataset.from_generator(DataGenerator(df, image_size).get_next_example,
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
                                                        'images': tf.TensorShape([5, image_size, image_size]),
                                                        'csky_ghi': tf.TensorShape([]),
                                                        'ghi': tf.TensorShape([])})

class DataGenerator(object):
    """
    Generator that yields one training example at a time.
    """
    def __init__(self, df : pd.DataFrame, image_size : int):
        self.df = df
        self.image_size = image_size
        self.stations = {'BND':(40.05192,-88.37309),
                         'TBL':(40.12498,-105.2368),
                         'DRA':(36.62373,-116.01947),
                         'FPK':(48.30783,-105.1017),
                         'GWN':(34.2547,-89.8729),
                         'PSU':(40.72012,-77.93085),
                         'SXF':(43.73403,-96.62328)}

    def get_next_example(self):
        # TODO : shuffle dataframe while keeping examples from the same file together (to improve performance)
        # TODO : Do not load night examples (with values ...__DAYTIME == 0)
        
        # Iterate over all rows of the dataframe
        for index, row in self.df.iterrows():
            hdf5_8bit_path = row['hdf5_8bit_path']
            hdf5_8bit_offset = row['hdf5_8bit_offset']
            
            # Open hdf5 file
            with h5py.File(hdf5_8bit_path, "r") as h5_data:
                
                # Get latitude & longitude stored in the file
                lats, lons = utils.fetch_hdf5_sample("lat", h5_data, hdf5_8bit_offset), utils.fetch_hdf5_sample("lon", h5_data, hdf5_8bit_offset)
                if lats is None or lons is None:
                    continue
                    
                # Get data from the 5 channels
                images = []
                for channel in ('ch1', 'ch2', 'ch3', 'ch4', 'ch6'):
                    img = utils.fetch_hdf5_sample(channel, h5_data, hdf5_8bit_offset)
                    if type(img) is np.ndarray:
                        images.append(img)
                if len(images) < 5:
                    continue
                
                # Yield relevant data for each station
                for station_name, station_coords in self.stations.items():
                    pixel_coords = (np.argmin(np.abs(lats - station_coords[0])), np.argmin(np.abs(lons - station_coords[1])))
                    ghi = row[station_name + "_GHI"]
                    csky_ghi = row[station_name + "_CLEARSKY_GHI"]
                    
                    # Crop the images with the station centered
                    pixels = self.image_size//2
                    adjustement = self.image_size % 2 # Adjustement if image_size is odd
                    cropped_images = []
                    for img in images:
                        # TODO : Check if the slice is out of bounds
                        cropped_images.append(img[pixel_coords[0]-pixels:pixel_coords[0]+pixels+adjustement,
                                          pixel_coords[1]-pixels:pixel_coords[1]+pixels+adjustement])
                    cropped_images = tf.convert_to_tensor(np.array(cropped_images))
                    
                    yield ({'hdf5_8bit_path': hdf5_8bit_path, 
                            'hdf5_8bit_offset': hdf5_8bit_offset,
                            'station_name': station_name,
                            'station_lat': pixel_coords[0],
                            'station_long': pixel_coords[1],
                            'images': cropped_images,
                            'csky_ghi': csky_ghi,
                            'ghi': ghi})


# ----- Code to benchmark dataset efficency ----- #
import time

def benchmark(dataset, num_examples=100):
    start_time = time.perf_counter()
    for i, sample in enumerate(dataset):
        time.sleep(0.01) # Simulating a training step
        if i >= num_examples:
            break
    tf.print("Execution time:", time.perf_counter() - start_time)
                    
if __name__ == "__main__":
    df = pd.read_pickle('/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl')
    benchmark(SolarIrradianceDataset(df,20)
    .prefetch(tf.data.experimental.AUTOTUNE))


