import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow debugging logs
import glob
import fire
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from models import baselines
from dataset.datasets import SolarIrradianceDataset
from dataset.crop_dataset import CropDataset
from dataset.sequence_dataset import SequenceDataset
from utils import preprocessing
from utils import utils
from utils import plots
from utils import logging
from utils import data

DATA_PATH = '/project/cq-training-1/project1/data/'
HDF5_8BIT = 'hdf5v7_8bit'
BATCH_LOG_INTERVAL = 50
VALID_PERC = 0.2
SLURM_TMPDIR = os.environ["SLURM_TMPDIR"] if "SLURM_TMPDIR" in os.environ else glob.glob('/localscratch/'+os.environ['USER']+'*')[0]

# Setup writers for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

logger = logging.get_logger()

# TODO : Probably move this in a data.py file in a data class (data.stations)
stations = {'BND':(40.05192,-88.37309), 'TBL':(40.12498,-105.2368),
            'DRA':(36.62373,-116.01947), 'FPK':(48.30783,-105.1017),
            'GWN':(34.2547,-89.8729), 'PSU':(40.72012,-77.93085), 
            'SXF':(43.73403,-96.62328)}

#@tf.function
def train_epoch(model, data_loader, batch_size, loss_function, optimizer):
    total_loss, nb_batch = 0, 0
    for batch in data_loader.batch(batch_size):
        images, labels = batch['images'], batch['ghi']
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = loss_function(y_true=labels, y_pred=preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        total_loss += loss
        nb_batch += 1
        
        # Tensorboard logging
        if nb_batch % BATCH_LOG_INTERVAL == 0: 
            with train_summary_writer.as_default():
                tf.summary.scalar('batch_loss_train', loss.numpy(), step=nb_batch)
                tf.summary.image(f"GHIs : {labels[0].numpy()}", np.moveaxis(images[0,-1,:,:,:, np.newaxis], -2, 0), step=nb_batch, max_outputs=5)

    return total_loss/nb_batch # Average total epoch loss

#@tf.function
def test_epoch(model, data_loader, batch_size, loss_function):
    total_loss, total_loss_csky, nb_batch = 0, 0, 0
    for batch in data_loader.batch(batch_size):
        images, labels, preds_csky = batch['images'], batch['ghi'], batch['csky_ghi']
        preds = model(images)
        total_loss += loss_function(y_true=labels, y_pred=preds)
        total_loss_csky += loss_function(y_true=labels, y_pred=preds_csky)
        nb_batch += 1
    return total_loss/nb_batch, total_loss_csky/nb_batch # Average total epoch loss

def main(df_path: str = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl', 
         image_size: int = 20,
         model: str = 'dummy',
         epochs: int = 10 ,
         optimizer: str = 'adam' ,
         lr: float = 1e-4 , 
         batch_size: int = 100,
         subset_perc: float = 1,
         saved_model_dir: str = None,
         seq_len: int = 6
        ):
    
    # Warning if no GPU detected
    if len(tf.config.list_physical_devices('GPU')) == 0:
        logger.warning('No GPU detected, training will run on CPU.')
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        logger.warning('Multiple GPUs detected, training will run on only one GPU.')

    # Load dataframe
    logger.info('Loading and preprocessing dataframe...')
    df = pd.read_pickle(df_path)
    df = preprocessing.preprocess(df, shuffle=False)

    # Split into train and valid
    df = df.iloc[:int(len(df.index)*subset_perc)]
    cutoff = int(len(df.index)*(1-VALID_PERC))
    df_train, df_valid = df.iloc[:cutoff], df.iloc[cutoff:]

    # Wrap dataframe
    ghis = data.GHIs(df)
    image_paths, image_paths_train, image_paths_valid = data.ImagePaths(df), data.ImagePaths(df_train), data.ImagePaths(df_valid)

    # Pre-crop data
    logger.info('Getting crops...')
    images = data.Images(image_paths, image_size)
    images.crop(dest=SLURM_TMPDIR)
    
    # Create model
    if model == 'dummy':
        model = baselines.DummyModel()
    elif model == 'sunset':
        model = baselines.SunsetModel()
    elif model == 'cnndem':
        model = baselines.ConvDemModel(image_size)
    elif model == 'sunset3d':
        model = baselines.Sunset3DModel(seq_len)
    else:
        raise Exception(f'Model \"{model}\" not recognized.')
        
    # Load model weights
    if saved_model_dir is not None:
        model.load_weights(os.path.join(saved_model_dir, "model"))
    
    # Loss and optimizer
    mse = tf.keras.losses.MeanSquaredError()
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise Exception(f'Optimizer \"{optimizer}\" not recognized.')
    
    if model.__class__.__name__ in ['Sunset3DModel']: # Temporary if to not break older models
        # Create data loader
        dataloader_train = SequenceDataset(image_paths_train, images, ghis, seq_len=seq_len)
        dataloader_valid = SequenceDataset(image_paths_valid, images, ghis, seq_len=seq_len)
    else:
        dataloader_train = SolarIrradianceDataset(df_train, image_size)
        dataloader_valid = SolarIrradianceDataset(df_valid, image_size)
    
    # Training loop
    logger.info('Training...')
    losses = {'train' : [], 'valid' : []}
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader_train, batch_size, mse, optimizer)
        valid_loss, csky_valid_loss = test_epoch(model, dataloader_valid, batch_size, mse)
        if valid_loss.numpy() < best_valid_loss:
            best_valid_loss = valid_loss.numpy()
            utils.save_model(model)
        
        # Logs
        logger.info(f'Epoch {epoch} - Train Loss : {train_loss:.4f}, Valid Loss : {valid_loss:.4f}')
        losses['train'].append(train_loss.numpy())
        losses['valid'].append(valid_loss.numpy())
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.numpy(), step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.numpy(), step=epoch)
            
    # Plot losses
    plots.plot_loss(losses['train'], losses['valid'], csky_valid_loss.numpy())
    
if __name__ == "__main__":
    fire.Fire(main)
