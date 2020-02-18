import datetime
import os
import numpy as np
import pandas as pd
import logging
import pickle
import shutil
import tarfile
import h5py
from utils import utils
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())
logger = logging.getLogger('logger')

CROP_PROCESSES_NB = 8

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
    def __init__(self, df, eight_bits=True):
        self.df = df
        self.col_path = 'hdf5_8bit_path' if eight_bits else 'hdf5_16bit_path'
        self.col_offset = 'hdf5_8bit_offset' if eight_bits else 'hdf5_16bit_path'
        self.col_csky = [s+'_CLEARSKY_GHI' for s in stations.keys()]
        self.col_ghi = [s+'_GHI' for s in stations.keys()]

    def ghis_exist(self, timestamp: datetime):
        return timestamp in self.df.index \
               and not self.df.loc[timestamp, self.col_ghi].isna().any() \
               and not self.df.loc[timestamp, self.col_csky].isna().any()

    def path_exist(self, timestamp: datetime):
        return timestamp in self.df.index \
                and self.df.loc[timestamp, 'ncdf_path'] is not np.NaN \
                and self.df.loc[timestamp, self.col_path] is not np.NaN \
                and self.df.loc[timestamp, self.col_offset] is not np.NaN

    def get_ghis(self, timestamp: datetime):
        if not self.ghis_exist(timestamp):
            empty_values = dict(zip(stations.keys(), [0]*len(stations.keys())))
            return empty_values, empty_values # TODO : Interpolate
        ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_ghi])))
        csky_ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_csky])))
        return ghis, csky_ghis

    def get_ghi(self, timestamp: datetime, station: str):
        ghis, csky_ghis = self.get_ghis(timestamp)
        return ghis[station], csky_ghis[station]

    def get_path(self, timestamp: datetime):
        if not self.path_exist(timestamp):
            return None
        return self.df.loc[timestamp, self.col_path], self.df.loc[timestamp, self.col_offset]

    def get_paths_subsets(self, n_subsets):
        df = self.df[['ncdf_path', self.col, self.offset]].dropna()
        df['just_date'] = df.index.date
        groups = [df for _, df in df.groupby('just_date')]
        cutoffs = np.arange(0, n_subsets+1)/n_subsets * len(groups)
        groups_subsets = [groups[int(cutoffs[i]):int(cutoffs[i+1])] for i in range(len(cutoffs)-1)]
        paths_subsets = []
        for groups in groups_subsets:
            paths = []
            df_subset = pd.concat(groups)
            for timestamp, row in df_subset.iterrows():
                paths.append((timestamp, row[self.col], row[self.offset]))
            paths_subsets.append(paths)
        return paths_subsets

    def is_night(self, timestamp: datetime, station: str):
        if timestamp not in self.df.index:
            logger.warning(f'is_night : could not find {timestamp}.')
            return True
        return self.df.loc[timestamp, station + '_DAYTIME'] == 0

    def get_timestamps(self):
        return self.df[['ncdf_path', self.col_path, self.col_offset] + self.col_csky + self.col_ghi].dropna().index

    def split(self, valid_perc=0.2):
        index = self.get_timestamps()
        cutoff = int(len(index)*(1-valid_perc))
        return Metadata(self.df.loc[index[:cutoff], :]), Metadata(self.df.loc[index[cutoff:], :])

    def __len__(self):
        return len(self.get_timestamps())

class GHIs(object):
    """
    Class that wraps the metadata dataframe to provide an interface on the GHIs values.
    """
    def __init__(self, df):
        self.df = df.replace('nan',np.NaN)
        self.col_csky = [s+'_CLEARSKY_GHI' for s in stations.keys()]
        self.col_ghi = [s+'_GHI' for s in stations.keys()]

    def get_ghis(self, timestamp: datetime):
        if not self.exists(timestamp):
            return 0, 0 # TODO : Interpolate
        ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_ghi])))
        csky_ghis = dict(zip(stations.keys(), list(self.df.loc[timestamp, self.col_csky])))
        return ghis, csky_ghis

    def get_ghi(self, timestamp: datetime, station: str):
        ghis, csky_ghis = self.get_ghis(timestamp)
        if type(ghis) == dict and type(csky_ghis) == dict:
            return ghis[station], csky_ghis[station]
        else:
            return ghis, csky_ghis
    
    def exists(self, timestamp: datetime):
        if timestamp not in self.df.index:
            #logger.warning(f'Timestamp \"{timestamp}\" not found in the metadata dataframe. Every timestamps are supposed to be there.')
            return False
        return not self.df.loc[timestamp, self.col_ghi].isna().any() and not self.df.loc[timestamp, self.col_csky].isna().any()

    def get_timestamps(self):
        return self.df.index

class ImagePaths(object):
    """
    Class that wraps the metadata dataframe to get image paths
    """
    def __init__(self, df, eight_bits=True):
        self.df = df.replace('nan',np.NaN)
        self.col = 'hdf5_8bit_path' if eight_bits else 'hdf5_16bit_path'
        self.offset = 'hdf5_8bit_offset' if eight_bits else 'hdf5_16bit_path'
    
    def get_path(self, timestamp: datetime):
        """Get path of the images for a given timestamp
        
        Arguments:
            timestamp {datetime} -- Timestamp corresponding to the desired image
        
        Returns:
            (string, int) -- Path and offset of the desired image if it exist or None
        """
        if not self.exists(timestamp):
            return None

        return self.df.loc[timestamp, self.col], self.df.loc[timestamp, self.offset]

    def exists(self, timestamp: datetime):
        """Checks if image exists for a given timestamp
        
        Arguments:
            timestamp {datetime} -- Timestamp corresponding to an image
        
        Returns:
            bool -- True if the image exist, False otherwise
        """
        if timestamp not in self.df.index:
            #logger.warning(f'Timestamp \"{timestamp}\" not found in the metadata dataframe. Every timestamps are supposed to be there.')
            return False

        return self.df.loc[timestamp, 'ncdf_path'] is not np.NaN \
                and self.df.loc[timestamp, self.col] is not np.NaN \
                and self.df.loc[timestamp, self.offset] is not np.NaN
    
    def is_night(self, timestamp: datetime, station: str):
        return self.df.loc[timestamp, station + '_DAYTIME'] == 0
    
    def yield_paths(self):
        for time in self.get_timestamps():
            if self.exists(time):
                yield time, self.get_path(time)

    def yield_time(self):
        for time in self.get_timestamps():
            if self.exists(time):
                yield time

    def get_timestamps(self):
        return self.df.index

    def get_paths(self):
        paths = []
        for timestamp, row in self.df[['ncdf_path', self.col, self.offset]].dropna().iterrows():
            paths.append((timestamp, row[self.col], row[self.offset]))
        return paths

    def get_paths_subsets(self, n_subsets):
        df = self.df[['ncdf_path', self.col, self.offset]].dropna()
        df['just_date'] = df.index.date
        groups = [df for _, df in df.groupby('just_date')]
        #print(f'length of groups : {len(groups)}')
        cutoffs = np.arange(0, n_subsets+1)/n_subsets * len(groups)
        #print(f'cutoffs : {cutoffs}')
        groups_subsets = [groups[int(cutoffs[i]):int(cutoffs[i+1])] for i in range(len(cutoffs)-1)]
        #print(f'length of groups_subset : {len(groups_subsets)}')
        #print(f'length of groups_subset[0] : {len(groups_subsets[0])}')


        paths_subsets = []
        for groups in groups_subsets:
            paths = []
            df_subset = pd.concat(groups)
            for timestamp, row in df_subset.iterrows():
                paths.append((timestamp, row[self.col], row[self.offset]))
            paths_subsets.append(paths)

        return paths_subsets

    def get_total_paths(self):
        return len(self.df[['ncdf_path', self.col, self.offset]].dropna())


class Images(object):
    """
    Class that reads and writes pre-cropped images on disk
    """
    def __init__(self, metadata: Metadata, image_size: int):
        self.metadata = metadata
        self.image_size = image_size
        self.shared_storage = '/project/cq-training-1/project1/teams/team12/'
        self.data_folder = None

    def get_images(self, timestamp: datetime):
        if self.data_folder is None:
            raise Exception('Data folder not set, call \"Images.crop\" before calling \"Images.get_images\".')
        file_path = os.path.join(self.data_folder, str(timestamp.date())+'.pkl')
        if not os.path.exists(file_path):
            return np.zeros((self.image_size, self.image_size, 5))
        with open(file_path, 'rb') as f:
            day_data = pickle.load(f)
        if timestamp not in day_data.keys():
            return np.zeros((self.image_size, self.image_size, 5))
        return day_data[timestamp]

    def get_image(self, timestamp: datetime, station: str):
        images = self.get_images(timestamp)
        if type(images) == dict:
            return images[station]
        return images

    def crop(self, dest: str):
        """Crops images and save them to destination. Only saves them to destination if images already exists on shared drive.
        
        Arguments:
            destination {str} -- Folder where to save the crops. Preferably on the compute node ($SLURM_TMPDIR)
        """
        def copy_tar(source, dest, crop_folder):
            logger.info(f'Copying file {source} to {dest}')
            shutil.copy(source,dest)
            logger.info('Extracting...')
            with tarfile.open(f'{os.path.join(dest,crop_folder)}.tar', 'r') as tarf:
                tarf.extractall(dest)
        
        # Crop only if destination folder do not exist
        crop_folder = f'crop-{self.image_size}'
        
        dest_folder = os.path.join(dest,crop_folder)
        shared = os.path.join(self.shared_storage, crop_folder)
        if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
            logger.info(f'Data already exist in destination {dest_folder}')
        elif os.path.exists(f'{shared}.tar'):
            copy_tar(f'{shared}.tar', dest, crop_folder)
        else:
            if not os.path.exists(shared):
                os.makedirs(shared)
            logger.info('Creating crops...')
            logger.info(f'Creating {CROP_PROCESSES_NB} threads...')
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
