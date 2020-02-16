import datetime
import numpy as np
import tensorflow as tf
from datetime import timedelta
import typing
from utils import data

class SequenceDataset(tf.data.Dataset):
    """
    Dataset that loads satellite imagery from pickle files.
    """    
    def __new__(cls, image_paths: data.ImagePaths, images: data.Images, ghis: data.GHIs, seq_len: int, timesteps: typing.Union[list, timedelta] = timedelta(minutes=15)):
        return tf.data.Dataset.from_generator(DataGenerator(image_paths, images, ghis, seq_len, timesteps).get_next_example,
                                            output_types={'station_name': tf.string,
                                                          'images': tf.float32,
                                                          'csky_ghi': tf.float32,
                                                          'ghi': tf.float32},
                                            output_shapes={'station_name': tf.TensorShape([]),
                                                           'images': tf.TensorShape([None, images.image_size, images.image_size, 5]),
                                                           'csky_ghi': tf.TensorShape([]),
                                                           'ghi': tf.TensorShape([])}).prefetch(tf.data.experimental.AUTOTUNE).cache()

class DataGenerator(object):
    """
    Generator that yields sequences of examples.
    """
    def __init__(self, image_paths: data.ImagePaths, images: data.Images, ghis: data.GHIs, seq_len: int, timesteps: typing.Union[list, timedelta]):
        self.image_paths = image_paths
        self.ghis = ghis
        self.images = images
        self.timesteps = [timesteps*i for i in range(seq_len)] if type(timesteps) == timedelta else timesteps

    def get_next_example(self):
        
        for time in self.image_paths.yield_time():
            
            for station in data.stations.keys():
                if self.image_paths.is_night(time, station):
                    continue
                
                # Get sequence
                img_seq = []
                for timestamp in [time - step for step in self.timesteps]:
                    img_seq.append(self.images.get_image(timestamp, station))
                
                # Get GHIs
                ghi_seq, csky_seq = [], []
                for timestamp in time + np.array([0, 1, 3, 6]) * timedelta(hours=1):
                    ghi, csky_ghi = self.ghis.get_ghi(timestamp, station)
                    ghi_seq.append(ghi)
                    csky_seq.append(csky_seq)

                yield ({'station_name': station,
                        'images': img_seq,
                        'csky_ghi': ghi_seq,
                        'ghi': csky_seq})
                


                
