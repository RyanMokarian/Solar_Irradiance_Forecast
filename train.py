import os
import fire
import datetime
import pandas as pd
from models import baselines
from utils.datasets import SolarIrradianceDataset
from utils import preprocessing
from utils import utils
from utils import plots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf

DATA_PATH = '/project/cq-training-1/project1/data/'
HDF5_8BIT = 'hdf5v7_8bit'
VALID_PERC = 0.2

# Setup writers for tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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
         copy_data: bool = False,
         saved_model_dir: str = None
        ):
    
    # Copy data to the compute node
    tmp_data_path = utils.copy_files(DATA_PATH, HDF5_8BIT) if copy_data else None

    # Load dataframe
    print('Loading and preprocessing dataframe...')
    df = pd.read_pickle(df_path)
    df = preprocessing.preprocess(df)
    
    # Split into train and valid
    df = df.iloc[:int(len(df.index)*subset_perc)]
    cutoff = int(len(df.index)*(1-VALID_PERC))
    df_train, df_valid = df.iloc[:cutoff], df.iloc[cutoff:]
    
    # Create model
    if model == 'dummy':
        model = baselines.DummyModel(image_size)
    elif model == 'another model name':
        pass # TODO : add new models here
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
    
    # Create data loader
    dataloader_train = SolarIrradianceDataset(df_train, image_size, tmp_data_path)
    dataloader_valid = SolarIrradianceDataset(df_valid, image_size, tmp_data_path)
    
    # Training loop
    print('Training...')
    losses = {'train' : [], 'valid' : []}
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader_train, batch_size, mse, optimizer)
        valid_loss, csky_valid_loss = test_epoch(model, dataloader_valid, batch_size, mse)
        if valid_loss.numpy() < best_valid_loss:
            best_valid_loss = valid_loss.numpy()
            utils.save_model(model)
        
        # Logs
        print(f'Epoch {epoch} - Train Loss : {train_loss:.4f}, Valid Loss : {valid_loss:.4f}')
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
