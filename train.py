import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow debugging logs
import glob
import fire
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import baselines
from models.cnn_gru.cnn_gru import CnnGru
from models.bi_lstm import LSTM_Resnet
from dataset.datasets import SolarIrradianceDataset
from dataset.sequence_dataset import SequenceDataset
from utils import preprocessing
from utils import utils
from utils import plots
from utils import logging
from utils import data

SEED = 1
DATA_PATH = '/project/cq-training-1/project1/data/'
HDF5_8BIT = 'hdf5v7_8bit'
BATCH_LOG_INTERVAL = 50
VALID_PERC = 0.1
#SLURM_TMPDIR = os.environ["SLURM_TMPDIR"] if "SLURM_TMPDIR" in os.environ else glob.glob('/localscratch/'+os.environ['USER']+'*')[0]
SLURM_TMPDIR = '/project/cq-training-1/project1/teams/team12'

# Setup writers for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

logger = logging.get_logger()

# Metrics
train_mse_metric = tf.keras.metrics.MeanSquaredError()
valid_mse_metric = tf.keras.metrics.MeanSquaredError()
valid_csky_mse_metric = tf.keras.metrics.MeanSquaredError()

def train_epoch(model, data_loader, batch_size ,loss_function, optimizer, total_examples, scale_label, use_csky):
    train_mse_metric.reset_states()
    for i, batch in tqdm(enumerate(data_loader), total=(np.ceil(total_examples/batch_size)), desc='train epoch', leave=False):
        images, labels, csky = batch['images'], batch['ghi'], batch['csky_ghi']
        with tf.GradientTape() as tape:
            preds = model(images,training=True)
            if use_csky:
                preds = preds + csky
            loss = loss_function(y_true=labels, y_pred=preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i%20 == 0:
            print(f'Predictions at {i}: {preds[0]}')

        if scale_label:
            preds, labels = preprocessing.unnormalize_ghi(preds), preprocessing.unnormalize_ghi(labels)

        train_mse_metric.update_state(y_true=labels, y_pred=preds)
        
        # Tensorboard logging
        if i % BATCH_LOG_INTERVAL == 0: 
            with train_summary_writer.as_default():
                tf.summary.image(f'Training data sample', np.moveaxis(images[0,-1,:,:,:, np.newaxis], -2, 0), step=i, max_outputs=5)

def test_epoch(model, data_loader, batch_size, loss_function, total_examples, scale_label, use_csky):
    valid_mse_metric.reset_states()
    valid_csky_mse_metric.reset_states()

    for batch in tqdm(data_loader, total=(np.ceil(total_examples/batch_size)), desc='valid epoch', leave=False):
        images, labels, csky = batch['images'], batch['ghi'], batch['csky_ghi']
        preds = model(images, training=False)
        if use_csky:
            preds = preds + csky
        if scale_label:
            preds, labels, csky = preprocessing.unnormalize_ghi(preds), preprocessing.unnormalize_ghi(labels), preprocessing.unnormalize_ghi(csky)
        
        valid_mse_metric.update_state(y_true=labels, y_pred=preds)
        valid_csky_mse_metric.update_state(y_true=labels, y_pred=csky)

def main(df_path: str = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl', 
         image_size: int = 32,
         model: str = 'dummy',
         epochs: int = 20,
         optimizer: str = 'adam' ,
         lr: float = 1e-4 , 
         batch_size: int = 32,
         subset_perc: float = 1,
         saved_model_dir: str = None,
         seq_len: int = 6,
         seed: bool = True,
         scale_label: bool = True,
         use_csky: bool = False,
         cache: bool = False, 
         timesteps_minutes: int = 30
        ):
    
    # Warning if no GPU detected
    if len(tf.config.list_physical_devices('GPU')) == 0:
        logger.warning('No GPU detected, training will run on CPU.')
    elif len(tf.config.list_physical_devices('GPU')) > 1:
        logger.warning('Multiple GPUs detected, training will run on only one GPU.')

    # Set random seed
    if seed:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

    # Load dataframe
    logger.info('Loading and preprocessing dataframe...')
    df = pd.read_pickle(df_path)
    df = preprocessing.preprocess(df, shuffle=False, scale_label=scale_label)
    metadata = data.Metadata(df, scale_label)

    # Pre-crop data
    logger.info('Getting crops...')
    images = data.Images(metadata, image_size)
    images.crop(dest=SLURM_TMPDIR)

    # Split into train and valid
    metadata, _ = metadata.split(1-subset_perc)
    metadata_train, metadata_valid = metadata.split(VALID_PERC)
    nb_train_examples, nb_valid_examples = metadata_train.get_number_of_examples(), metadata_valid.get_number_of_examples()
    logger.info(f'Number of training examples : {nb_train_examples}, number of validation examples : {nb_valid_examples}')

    # Create model
    if model == 'dummy':
        model = baselines.DummyModel()
    elif model == 'sunset':
        model = baselines.SunsetModel()
    elif model == 'cnndem':
        model = baselines.ConvDemModel(image_size)
    elif model == 'sunset3d':
        model = baselines.Sunset3DModel()
    elif model == 'convlstm':
        model = baselines.ConvLSTM()
    elif model == 'cnngru':
        model = CnnGru(seq_len)
    elif model == 'cnnlstm':
        model = LSTM_Resnet(seq_len)
    else:
        raise Exception(f'Model "{model}" not recognized.')
        
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
        raise Exception(f'Optimizer "{optimizer}" not recognized.')
    
    # Create data loader
    dataloader_train = SequenceDataset(metadata_train, images, seq_len=seq_len, 
    timesteps=datetime.timedelta(minutes=timesteps_minutes), cache=cache).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    dataloader_valid = SequenceDataset(metadata_valid, images, seq_len=seq_len, 
    timesteps=datetime.timedelta(minutes=timesteps_minutes), cache=cache).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    
    # Training loop
    logger.info('Training...')
    losses = {'train' : [], 'valid' : []}
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        train_epoch(model, dataloader_train, batch_size, mse, optimizer, nb_train_examples, scale_label, use_csky)
        test_epoch(model, dataloader_valid, batch_size ,mse, nb_valid_examples, scale_label, use_csky)
        train_loss = np.sqrt(train_mse_metric.result().numpy())
        valid_loss = np.sqrt(valid_mse_metric.result().numpy())
        csky_valid_loss = np.sqrt(valid_csky_mse_metric.result().numpy())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            utils.save_model(model)
        
        # Logs
        logger.info(f'Epoch {epoch} - Train Loss : {train_loss:.4f}, Valid Loss : {valid_loss:.4f}, Csky Valid Loss : {csky_valid_loss:.4f}' )
        losses['train'].append(train_loss)
        losses['valid'].append(valid_loss)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss, step=epoch)
            
    # Plot losses
    plots.plot_loss(losses['train'], losses['valid'], csky_valid_loss)
    
if __name__ == "__main__":
    fire.Fire(main)
