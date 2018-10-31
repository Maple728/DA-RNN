import numpy as np

from configs.config import *
from preprocess.get_data import *
from models.DA_RNN import *


MODEL_PATH = './checkpoints/TaxiNYConfig_rnn.ckpt'
is_train = True
RMSE_STOP_THRESHOLD = 0.0001


# --------------------- Data Process -----------------------
config = TaxiNYConfig
ds_handler = TaxiNYDataset(config)

dataset = ds_handler.get_dataset()
y_scaler = dataset[-1, -1, 0]

train_ds, valid_ds, test_ds = ds_handler.divide_three_ds(dataset)

model = DARNN(config)
sess = tf.Session()
saver = tf.train.Saver()

if is_train:
    sess.run(tf.global_variables_initializer())
    
    last_rmse = 100.0
    best_valid_rmse = 100.0
    for i in range(1000):

        batch_data = ds_handler.get_batch_data(train_ds)
        loss, mape, rmse = model.train(batch_data, y_scaler, sess)
        print('Epoch', i, 'Train Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
        if abs(last_rmse - rmse) < RMSE_STOP_THRESHOLD:
            break
        last_rmse = rmse
        
        if i % 10 == 0:
            batch_data = ds_handler.get_batch_data(valid_ds)
            loss, mape, rmse = model.predict(batch_data, y_scaler, sess)
            print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)
            if best_valid_rmse > rmse:
                best_valid_rmse = rmse
                # save model
                saver.save(sess, MODEL_PATH)

    print('Training completed')
    print('Best valid rmse:', best_valid_rmse)

else:
    saver.restore(sess, MODEL_PATH)

    batch_data = ds_handler.get_batch_data(valid_ds)
    loss, mape, rmse = model.predict(batch_data, y_scaler, sess)
    print('Valid Loss:', loss, 'MAPE(%):', mape * 100, 'RMSE:', rmse)    
