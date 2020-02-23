import datetime
import numpy as np
import tensorflow as tf
import typing
import logging
from datetime import timedelta
from utils import data

logger = logging.getLogger('logger')

class SequenceDataset(tf.data.Dataset):
    """
    Dataset that loads satellite imagery from pickle files.
    """    
    def __new__(cls, metadata: data.Metadata, images: data.Images, seq_len: int, batch_size:int ,timesteps: typing.Union[list, timedelta] = timedelta(minutes=15), cache: bool = False):
        dataset = tf.data.Dataset.from_generator(DataGenerator(metadata, images, seq_len, timesteps).get_next_example,
                                            output_types={'station_name': tf.string,
                                                          'images': tf.float32,
                                                          'csky_ghi': tf.float32,
                                                          'ghi': tf.float32,
                                                          'enc_stamps':tf.int16,
                                                          'seq_c':tf.float32},
                                            output_shapes={'station_name': tf.TensorShape([]),
                                                           'images': tf.TensorShape([None, images.image_size, images.image_size, 5]),
                                                           'csky_ghi': tf.TensorShape([4]),
                                                           'ghi': tf.TensorShape([4]),
                                                           'enc_stamps':tf.TensorShape([None,4]),
                                                           'seq_c':tf.TensorShape([None])}).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        if cache:
            return dataset.cache()
        else:
            return dataset

class DataGenerator(object):
    """
    Generator that yields sequences of examples.
    """
    def __init__(self, metadata: data.Metadata, images: data.Images, seq_len: int, timesteps: typing.Union[list, timedelta]):
        self.metadata = metadata
        self.images = images
        self.timesteps = [timesteps*i for i in range(seq_len)] if type(timesteps) == timedelta else timesteps
        # Present images as To-delta ..to.. To
        self.timesteps.reverse()

    def get_next_example(self):
        
        for time in self.metadata.get_timestamps():

            for station in data.stations.keys():
                if self.metadata.is_night(time, station):
                    continue
                
                # Get sequence
                img_seq, enc_stamps, seq_c = [], [], []
                for timestamp in [time - step for step in self.timesteps]:
                    img_seq.append(self.images.get_image(timestamp, station))
                    enc_stamps.append(self.metadata.enc_timestamps(timestamp))
                    seq_c.append(self.metadata.get_clearsky(timestamp,station))
        
                # Get GHIs
                ghi_seq, csky_seq = [], []
                for timestamp in time + np.array([0, 1, 3, 6]) * timedelta(hours=1):
                    ghi, csky_ghi = self.metadata.get_ghi(timestamp, station)
                    ghi_seq.append(ghi)
                    csky_seq.append(csky_ghi)
                    enc_stamps.append(self.metadata.enc_timestamps(timestamp))

                yield ({'station_name': station,
                        'images': img_seq,
                        'csky_ghi': csky_seq,
                        'ghi': ghi_seq,
                        'enc_stamps':enc_stamps,
                        'seq_c':seq_c})
                
