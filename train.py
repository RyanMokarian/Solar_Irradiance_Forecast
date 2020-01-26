import fire
import pandas as pd
import tensorflow as tf
from models import baselines
from utils.datasets import SolarIrradianceDataset
from utils import preprocessing

def main(df_path: str = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl' , 
         image_size: int = 20,
         model: str = 'dummy',
         epochs: int = 10 ,
         optimizer: str = 'adam' ,
         lr: float = 1e-4 , 
         batch_size: int = 2):

    # Load dataframe
    df = pd.read_pickle(df_path)
    df = preprocessing.preprocess(df)
    # TODO : Split 'df' into 'df_train' & 'df_valid' 
    
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
    data_loader = SolarIrradianceDataset(df, image_size)
    # TODO : Create a train data_loader and valid data_loader
    
    # Training loop
    for epoch in range(epochs):
        for batch in data_loader.batch(batch_size).take(2): # TODO : REMOVE .take() when training for real
            images = batch['images']
            label = batch['ghi']
            with tf.GradientTape() as tape:
                preds = model(images)
                #preds = batch['csky_ghi']
                loss = mse(y_true=label, y_pred=preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'Epoch {epoch} - Loss : {loss}')
    
if __name__ == "__main__":
    fire.Fire(main)
