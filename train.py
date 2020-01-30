import fire
import tqdm as tqdm
import pandas as pd
import tensorflow as tf
from models import baselines
from utils.datasets import SolarIrradianceDataset
from utils import preprocessing

DATA_PATH = '/project/cq-training-1/project1/data/'
VALID_PERC = 0.2

def setup():
    # TODO : Disable too much logging from tensorflow
    # TODO : Copy files to $SLURM
    pass

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
         subset_perc: float = 1
        ):
    
    # Setup the environment 
    setup()

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
        model = baselines.DummyModel()
    elif model == 'another model name':
        pass # TODO : add new models here
    else:
        raise Exception(f'Model \"{model}\" not recognized.')
    
    # Loss and optimizer
    mse = tf.keras.losses.MeanSquaredError()
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise Exception(f'Optimizer \"{optimizer}\" not recognized.')
    
    # Create data loader
    dataloader_train = SolarIrradianceDataset(df_train, image_size)
    dataloader_valid = SolarIrradianceDataset(df_valid, image_size)
    
    # Training loop
    print('Starting training...')
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader_train, batch_size, mse, optimizer)
        valid_loss, csky_valid_loss = test_epoch(model, dataloader_valid, batch_size, mse)
        print(f'Epoch {epoch} - Train Loss : {train_loss}, Valid Loss : {valid_loss}')
    
if __name__ == "__main__":
    fire.Fire(main)
