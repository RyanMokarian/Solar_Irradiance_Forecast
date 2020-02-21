import numpy as np
import typing
import datetime

import tensorflow as tf
import h5py

from datetime import timedelta
from utils import utils
from utils import data
from utils import logging

logger = logging.get_logger()

class EvaluatorDataset(tf.data.Dataset):
    """Dataset that loads satellite imagery from HDF5 files."""

    def __new__(cls, metadata: data.Metadata, image_size: int, seq_len: int, target_datetimes: list, stations: dict, target_time_offsets: list, timesteps: typing.Union[list, timedelta] = timedelta(minutes=15), batch=10):
        return tf.data.Dataset.from_generator(DataGenerator(metadata, image_size, seq_len, timesteps, target_datetimes, stations, target_time_offsets).get_next_example,
                                              output_types=(tf.float32, tf.float32)).prefetch(tf.data.experimental.AUTOTUNE).batch(batch)

class DataGenerator(object):
    """Generator that yields sequences of examples."""

    def __init__(self, metadata: data.Metadata, image_size: int, seq_len: int, timesteps: typing.Union[list, timedelta], target_datetimes: list, stations: dict, target_time_offsets: list):
        self.metadata = metadata
        self.target_datetimes = target_datetimes
        self.stations = stations
        self.target_time_offsets = target_time_offsets
        self.timesteps = [timesteps*i for i in range(seq_len)] if type(timesteps) == timedelta else timesteps
        self.image_reader = ImageReader(self.metadata, image_size)

    def get_next_example(self):

        for time in self.target_datetimes : 
            for station_name, coords in self.stations.items():

                # Get sequence
                img_seq = []
                for timestamp in [time - step for step in self.timesteps]:
                    img_seq.append(self.image_reader.get_image(timestamp, coords))
                
                # Get ClearSky GHIs
                csky_seq = []
                for timestamp in time + np.array(self.target_time_offsets):
                    csky_ghi = self.metadata.get_clearsky(timestamp, station_name)
                    csky_seq.append(csky_ghi)

                yield np.array(img_seq), None # Targets are None because they are not available for the evaluation.

class ImageReader(object):
    """Class that reads images encoded in HDF5 files."""
    def __init__(self, metadata: data.Metadata, image_size: int):
        self.metadata = metadata
        self.image_size = image_size
        self.cache = None

    def get_image(self, timestamp: datetime, station_coords: tuple):
        path_offset = self.metadata.get_path(timestamp)
        if path_offset is None:
            #logger.warning(f'{timestamp} is unavailable. Returning empty image.')
            return np.zeros((self.image_size, self.image_size, 5))
        path, offset = path_offset
        h5_data = h5py.File(path, "r")

        # Get latitude & longitude stored in the file
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, offset), utils.fetch_hdf5_sample("lon", h5_data, offset)
        if lats is None or lons is None:
            #logger.warning(f'{timestamp} is unavailable. Returning empty image.')
            return np.zeros((self.image_size, self.image_size, 5))
                
        # Get data from the 5 channels
        images = []
        for channel in ('ch1', 'ch2', 'ch3', 'ch4', 'ch6'):
            img = utils.fetch_hdf5_sample(channel, h5_data, offset)
            if type(img) is np.ndarray:
                images.append(img)
            else:
                #logger.warning(f'Channel "{channel}" is not available at date {timestamp}, it will be zeros.')
                images.append(np.zeros((self.image_size, self.image_size)))
            
        # Crop image
        pixel_coords = (np.argmin(np.abs(lats - station_coords[0])), np.argmin(np.abs(lons - station_coords[1])))
        pixels = self.image_size//2
        adjustement = self.image_size % 2 # Adjustement if image_size is odd
        cropped_images = []
        for img, mean, std in zip(images, data.images_mean.values(), data.images_std.values()):
            # TODO : Check if the slice is out of bounds
            img = (img - mean)/std # Normalize image
            cropped_images.append(img[pixel_coords[0]-pixels:pixel_coords[0]+pixels+adjustement,
                                pixel_coords[1]-pixels:pixel_coords[1]+pixels+adjustement])
        return np.moveaxis(np.array(cropped_images), 0, -1)

