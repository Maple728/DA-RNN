import tensorflow as tf

from configs.config import *
from preprocess.get_data import *
from models.DA_RNN import *

# --------------------- Setting -----------------
RMSE_STOP_THRESHOLD = 0.0001
# TaxiNYDataset, NasdaqDataset
DATASET_CLASS = NasdaqDataset
# TaxiNYConfig, NasdaqConfig
config = NasdaqConfig


def train_config(config):
    with tf.Session() as sess:
        # data process
        ds_handler = DATASET_CLASS(config)
        dataset = ds_handler.get_dataset()
        y_scaler = dataset[-1, -1, 0]
        train_ds, valid_ds, test_ds = ds_handler.divide_three_ds(dataset)
        
        # train
        model = DARNN(config)
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        last_rmse = 100.0
        best_valid_rmse = 100.0
        for i in range(1000):
            train_batch_ds = ds_handler.get_batch_data(train_ds)
            
            loss, mape, rmse = model.train(train_batch_ds, y_scaler, sess)
            print('Epoch', i, 'Train Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
            
            if abs(last_rmse - rmse) < RMSE_STOP_THRESHOLD:
                last_rmse = rmse
                break
            last_rmse = rmse
            
            if i % 10 == 0:
                valid_batch_ds = ds_handler.get_batch_data(valid_ds)
                loss, mape, rmse = model.predict(valid_batch_ds, y_scaler, sess)
                print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
                if best_valid_rmse > rmse:
                    best_valid_rmse = rmse

        print('Training completed')
        print('-------------Best valid rmse:', best_valid_rmse)

        return config.hidden_sizes, config.timestep, best_valid_rmse

# --------------------- main run ---------------------------
train_results = []
for hidden_sizes in [ [64], [128], [256]]:
    config.hidden_sizes = hidden_sizes
    for timestep in [6, 10, 12]:
        config.timestep = timestep
        
        print('!!!Train config, hidden sizes:', hidden_sizes, 'timestep:', timestep)
        train_results.append(train_config(config))

        # reset graph to free graph and memory
        tf.reset_default_graph()

train_results = sorted(train_results, key = lambda item : item[-1])

print('---------------- Summary ---------')
print('Hidden_sizes, timestep, RMSE')
for record in train_results:
    print(record)
