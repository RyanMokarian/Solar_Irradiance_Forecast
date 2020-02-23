
import os
import numpy as np
import pandas as pd
import logging
import pickle
import shutil
import tarfile
import h5py
from datetime import datetime
from utils import utils
from tqdm import tqdm
from multiprocessing import Pool
from collections import deque
logger = logging.getLogger('logger')

CROP_PROCESSES_NB = 4

GHI_MEDIAN = 291.1266666666796
GHI_MEAN = 357.8474970021783
GHI_STD = 293.66987323582606

stations = {'BND':(40.05192,-88.37309), 'TBL':(40.12498,-105.2368),
            'DRA':(36.62373,-116.01947), 'FPK':(48.30783,-105.1017),
            'GWN':(34.2547,-89.8729), 'PSU':(40.72012,-77.93085), 
            'SXF':(43.73403,-96.62328)}

images_mean = {'ch1': 0.3154958181516178, 'ch2': 274.18017705595855, 
                    'ch3': 230.33078484969695, 'ch4': 264.2658065947056, 
                    'ch6': 245.69855600075982}
                    
images_std = {'ch1': 0.30688239459685107, 'ch2': 60.6773046722939, 
                    'ch3': 50.38130188316295, 'ch4': 59.25418221268753, 
                    'ch6': 54.37427451580163}

class Metadata(object):
    """Wrapper class to handle metadata dataframe."""

    def __init__(self, df, scale_label, eight_bits=True):
        self.df = df
        self.col_path = 'hdf5_8bit_path' if eight_bits else 'hdf5_16bit_path'
        self.col_offset = 'hdf5_8bit_offset' if eight_bits else 'hdf5_16bit_path'
        self.col_csky = [s+'_CLEARSKY_GHI' for s in stations.keys()]
        self.col_ghi = [s+'_GHI' for s in stations.keys()]
        self.col_daytime = [s+'_DAYTIME' for s in stations.keys()]
        self.scale_label = scale_label
        self.ghi_median = (GHI_MEDIAN-GHI_MEAN)/GHI_STD if self.scale_label else GHI_MEDIAN

    def ghis_exist(self, timestamp: datetime):
        """Checks if GHI values exist in the dataframe for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp for which to check.
        
        Returns:
            bool -- True if the GHIs exist, False otherwise.
        """
        return timestamp in self.df.index \
               and not self.df.loc[timestamp, self.col_ghi].isna().any() \
               and not self.df.loc[timestamp, self.col_csky].isna().any()

    def path_exist(self, timestamp: datetime):
        """Checks if a path exist in the dataframe for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp for which to check.
        
        Returns:
            bool -- True if the path exist, False otherwise.
        """
        return timestamp in self.df.index \
                and self.df.loc[timestamp, 'ncdf_path'] is not np.NaN \
                and self.df.loc[timestamp, self.col_path] is not np.NaN \
                and self.df.loc[timestamp, self.col_offset] is not np.NaN

    def get_ghis(self, timestamp: datetime):
        """Gets the GHIs for all stations for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp of the GHIs.
        
        Returns:
            dict -- dict were keys are the station names and values are the GHIs.
        """
        if not self.ghis_exist(timestamp):
            
            median_values = dict(zip(stations.keys(), [self.ghi_median]*len(stations.keys())))
            return median_values, median_values
        ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_ghi])))
        csky_ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_csky])))
        return ghis, csky_ghis

    def get_ghi(self, timestamp: datetime, station: str):
        """Gets the GHI for one station for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp of the requested GHI.
            station {str} -- The station of the requested GHI.
        
        Returns:
            tuple -- GHI, Clearsky GHI
        """
        if timestamp in self.df.index:
            return self.df.loc[timestamp, station+'_GHI'], self.df.loc[timestamp, station+'_CLEARSKY_GHI']
        else:
            return self.ghi_median, self.ghi_median

    def get_clearsky(self, timestamp: datetime, station: str):
        """Gets the Clearsky GHI ONLY. This method is used in the evaluator dataset when
        the labels are unavailable
        
        Arguments:
            timestamp {datetime} -- The timestamp of the requested Clearsky GHI.
            station {str} -- The station of the requested Clearsky GHI.

        Returns:
            float -- Clearsky GHI
        """
        if timestamp in self.df.index and not self.df.loc[timestamp, self.col_csky].isna().any():
            return self.df.loc[timestamp, station+'_CLEARSKY_GHI']
        else:
            return self.ghi_median

    def get_path(self, timestamp: datetime):
        """Gets the path of the hdf5 file for a particular datetime.
        
        Arguments:
            timestamp {datetime} -- The timestamp of the required path.
        
        Returns:
            tuple -- Path, Offset
        """
        if not self.path_exist(timestamp):
            return None
        return self.df.loc[timestamp, self.col_path], self.df.loc[timestamp, self.col_offset]

    def get_paths_subsets(self, n_subsets):
        """Divides the paths into n_subsets. This is useful to process data using n threads.
        
        Arguments:
            n_subsets {int} -- Number of subsets
        
        Returns:
            list -- List of list of paths.
        """
        df = self.df[['ncdf_path', self.col_path, self.col_offset]].dropna()

        # Group data by day (each day is saved in one file)
        df['just_date'] = df.index.date
        groups = [df for _, df in df.groupby('just_date')]

        # Divide the days into n_subsets sets
        cutoffs = np.arange(0, n_subsets+1)/n_subsets * len(groups)
        groups_subsets = [groups[int(cutoffs[i]):int(cutoffs[i+1])] for i in range(len(cutoffs)-1)]

        # Create a list of tuple (time, path, offset)
        paths_subsets = []
        for groups in groups_subsets:
            paths = []
            df_subset = pd.concat(groups)
            for timestamp, row in df_subset.iterrows():
                paths.append((timestamp, row[self.col_path], row[self.col_offset]))
            paths_subsets.append(paths)

        return paths_subsets

    def is_night(self, timestamp: datetime, station: str):
        """Checks if it is night at a given time and station.
        
        Arguments:
            timestamp {datetime} -- The timestamp for which to check.
            station {str} -- The station for wich to check.
        
        Returns:
            bool -- True if it is night, False otherwise.
        """
        if timestamp not in self.df.index:
            logger.warning(f'is_night : could not find {timestamp}.')
            return True
        return self.df.loc[timestamp, station + '_DAYTIME'] == 0

    def get_timestamps(self):
        """Gets the valid timestamps.
        
        Returns:
            list -- List of valid timestamps.
        """
        return self.df[['ncdf_path', self.col_path, self.col_offset] + self.col_csky + self.col_ghi].dropna().index

    def split(self, valid_perc: float = 0.2):
        """Splits the data into a training and validation set.
        
        Keyword Arguments:
            valid_perc {float} -- Percentage of the data to use in the validation set. (default: {0.2})
        
        Returns:
            tuple -- Tuple of data.Metadata objects. Respectively train and validation.
        """
        index = self.get_timestamps()
        cutoff = int(len(index)*(1-valid_perc))
        return Metadata(self.df.loc[index[:cutoff]], self.scale_label), Metadata(self.df.loc[index[cutoff:]], self.scale_label)

    def split_with_dates(self, dates: list = ['2013-01','2014-06','2012-08','2011-03','2010-10']):
        """Splits the data into a training and validation set. Uses the dates in the validation set.
        
        Arguments:
            dates {list} -- List of string dates in the YYYYMM format. (default: {['2013-01','2014-06','2012-08','2011-03','2010-10']})

        Returns:
            tuple -- Tuple of data.Metadata objects. Respectively train and validation.
        """
        train_idx, valid_idx = [], []
        for date in dates:
            if len(date) == 4:
                for index in self.get_timestamps():
                    if date == str(index.year):
                        valid_idx.append(index)
                    else:
                        train_idx.append(index)
            elif len(date) == 7:
                year, month = date.split('-')
                for index in self.get_timestamps():
                    if year == str(index.year) and month == str(index.month):
                        valid_idx.append(index)
                    else:
                        train_idx.append(index)
            else:
                logger.error(f'Date format not recognised : {date}')
            
        return Metadata(self.df.loc[train_idx], self.scale_label), Metadata(self.df.loc[valid_idx], self.scale_label)
    
    def get_number_of_examples(self):
        """Gets the total amount of valid examples"""
        return self.df.loc[self.get_timestamps(), self.col_daytime].values.sum()
    
    def enc_timestamps(self,stamp):
        """One hot encoding elements of the timestep"""
        return [stamp.month-1,stamp.day-1,stamp.hour,[0,15,30,45].index(stamp.minute)]

    def __len__(self):
        return len(self.get_timestamps())

class Images(object):
    """
    Class that reads and writes pre-cropped images on disk
    """
    def __init__(self, metadata: Metadata, image_size: int):
        self.metadata = metadata
        self.image_size = image_size
        self.shared_storage = '/project/cq-training-1/project1/teams/team12/'
        self.data_folder = None
        self.cache = {}

    def get_images(self, timestamp: datetime):
        """Gets the images for all stations for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp of the images.
        
        Returns:
            dict -- dict were keys are the station names and values are the images.
        """
        if self.data_folder is None:
            raise Exception('Data folder not set, call "Images.crop" before calling "Images.get_images".')
        
        # Check if image is in the cache
        if timestamp in self.cache.keys():
            return self.cache[timestamp]
        
        # Check if the image is saved on disk
        file_path = os.path.join(self.data_folder, str(timestamp.date())+'.pkl')
        if not os.path.exists(file_path):
            return np.zeros((self.image_size, self.image_size, 5))
        
        # Read the file and update the cache
        with open(file_path, 'rb') as f:
            day_data = pickle.load(f)
        self.cache = day_data

        # Check that the image is in file
        if timestamp not in day_data.keys():
            return np.zeros((self.image_size, self.image_size, 5))

        return day_data[timestamp]

    def get_image(self, timestamp: datetime, station: str):
        """Gets the channels for one stations for a particular timestamp.
        
        Arguments:
            timestamp {datetime} -- The timestamp of the requested image.
            station {str} -- The station of the requested image.
        
        Returns:
            np.array -- Channels of the requested image.
        """
        images = self.get_images(timestamp)
        if type(images) == dict:
            return images[station]
        return images

    def crop(self, dest: str):
        """Crops images and save them to destination.
        
        Arguments:
            destination {str} -- Folder where to save the crops. Preferably on the compute node ($SLURM_TMPDIR)
        """
        def copy_tar(source, dest, crop_folder):
            logger.info(f'Copying file {source} to {dest}')
            shutil.copy(source,dest)
            logger.info('Extracting...')
            with tarfile.open(f'{os.path.join(dest,crop_folder)}.tar', 'r') as tarf:
                tarf.extractall(dest)
        
        crop_folder = f'crop-{self.image_size}'
        dest_folder = os.path.join(dest,crop_folder)
        shared = os.path.join(self.shared_storage, crop_folder)

        if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0: 
            logger.info(f'Data already exist in destination {dest_folder}')
        elif os.path.exists(f'{shared}.tar'):
            copy_tar(f'{shared}.tar', dest, crop_folder)
        else: # Crop only if destination folder do not exist
            if not os.path.exists(shared):
                os.makedirs(shared)

            # Cropping data
            logger.info(f'Creating {CROP_PROCESSES_NB} threads to crop data...')
            p = Pool(CROP_PROCESSES_NB)
            logger.info('Dividing task...')
            paths_subsets = self.metadata.get_paths_subsets(CROP_PROCESSES_NB)
            args = list(zip([shared]*CROP_PROCESSES_NB, paths_subsets, [self.image_size]*CROP_PROCESSES_NB, list(range(CROP_PROCESSES_NB))))
            logger.info(f'Sending task to {CROP_PROCESSES_NB} threads...')
            p.map(create_crops, args)

            logger.info('Taring crops...')
            with tarfile.open(f'{shared}.tar','w') as tarf:
                tarf.add(shared,crop_folder)
            copy_tar(f'{shared}.tar', dest, crop_folder)
        self.data_folder = dest_folder


# Function need to be defined at the top-level of the module for multiprocessing
def create_crops(args: tuple):
    """Function executed by multiple threads to create crops around stations 
        and save them to disk as pickle.
    
    Arguments:
        args {tuple} -- dest, paths, image_size, thread_nb
    """
    dest, paths, image_size, thread_nb = args
    # Iterate over all the existing timestamps
    open_path = None
    curr_date = None
    for time, path, offset in tqdm(paths, position=thread_nb, desc=f'Thread {thread_nb}', leave=False):
        # Open hdf5 file if it is not already opened
        if open_path != path:
            if open_path != None:
                h5_data.close()
            h5_data = h5py.File(path, "r")
            open_path = path

        # Save cropped images
        if curr_date != time.date():
            if curr_date is not None:
                with open(os.path.join(dest, str(curr_date)+'.pkl'), 'wb') as f:
                    pickle.dump(cropped_images_day, f) 
            cropped_images_day = {}
            curr_date = time.date()
        
        # Get latitude & longitude stored in the file
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, offset), utils.fetch_hdf5_sample("lon", h5_data, offset)
        if lats is None or lons is None:
            logger.warning(f'latlong of date {time} is unavailable, skipping...')
            continue
                
        # Get data from the 5 channels
        images = []
        for channel in ('ch1', 'ch2', 'ch3', 'ch4', 'ch6'):
            img = utils.fetch_hdf5_sample(channel, h5_data, offset)
            if type(img) is np.ndarray:
                images.append(img)
        if len(images) < 5:
            logger.warning(f'{5-len(images)} channels are not available at date {index}, skipping...')
            continue
            
        # Crop stations
        cropped_images_stations = {}
        for station_name, station_coords in stations.items():
            pixel_coords = (np.argmin(np.abs(lats - station_coords[0])), np.argmin(np.abs(lons - station_coords[1])))
                
            # Crop the images with the station centered
            pixels = image_size//2
            adjustement = image_size % 2 # Adjustement if image_size is odd
            cropped_images = []
            for img, mean, std in zip(images, images_mean.values(), images_std.values()):
                # TODO : Check if the slice is out of bounds
                img = (img - mean)/std # Normalize image
                cropped_images.append(img[pixel_coords[0]-pixels:pixel_coords[0]+pixels+adjustement,
                                    pixel_coords[1]-pixels:pixel_coords[1]+pixels+adjustement])
            cropped_images_stations[station_name] = np.moveaxis(np.array(cropped_images), 0, -1)
        cropped_images_day[time] = cropped_images_stations

    # Save the last day
    if len(cropped_images_day.keys()) > 0:
        with open(os.path.join(dest, str(time.date())+'.pkl'), 'wb') as f:
            pickle.dump(cropped_images_day, f)
