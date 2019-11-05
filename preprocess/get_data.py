import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NasdaqDataset:
    name = 'Nasdaq'
    data_filename = './dataset/nasdaq100_padding.csv'
    
    config = None
    def __init__(self, config):
        self.config = config
        
    def divide_three_ds(self, dataset):
        config = self.config
        #train_end_index = int(dataset.shape[1] / 10 * 8)
        #valid_end_index = int(dataset.shape[1] / 10 * 9)
        train_end_index = 35100
        valid_end_index = 37830  

        return dataset[ :,  : train_end_index], dataset[:,  train_end_index - config.timestep : valid_end_index], dataset[:, valid_end_index - config.timestep : -1]

    '''
        Last element is scaler if isScaled is true.
        Return <stock size, data point size + 1, 1>
    '''
    def get_dataset(self):
        config = self.config
        tmp_df = pd.read_csv(self.data_filename)
        data_list = np.array(tmp_df)
        # <stock, quotes>
        data = data_list.transpose()
        data = data.reshape( (data.shape[0], data.shape[1], 1) )
        if config.is_scaled:
            data = data.tolist()
            for i in range(len(data)):
                scaler = MinMaxScaler( config.feature_range )
                data[i] = scaler.fit_transform(data[i]).tolist()
                data[i].append( [scaler] )
            return np.array(data)
        else:
            return data
        

    def get_batch_data(self, data):
        config = self.config
        batch_data = []
        for i in range(config.timestep, data.shape[-2]):
            batch_data.append(data[:, i - config.timestep : i])
            if len(batch_data) == config.batch_size:
                # <batch_size, config.n + 1, config.timestep, 1>
                batch_data = np.array(batch_data)
                yield batch_data[:, : -1, :, :], batch_data[:, -1, : -1, :], batch_data[:, -1, -1, :]

                batch_data = []

class TaxiNYDataset:
    name = 'TaxiNY'
    time_interval = 15
    data_filename = './dataset/grid_map_dict_%smin.pickle' % time_interval

    def __init__(self, config):
        self.config = config

    '''
        Last element is scaler if isScaled is true.
        Return <grid map size, data point size + 1, 1> if is_scaled is True, otherwise <grid map size, data point size, 1> 
    '''
    def get_dataset(self):
        config = self.config
        # get data and process shape
        # <n, lat_len, lon_len>
        with open(self.data_filename, 'rb') as f:
            grid_map_dict = pickle.load(f)

        # transform dict to grid map list ordered by key ascending
        # sort asc
        grid_map_list = sorted(grid_map_dict.items(), key = lambda item : item[0])
        grid_map_list = list(map(lambda item : item[1], grid_map_list))
        
        # transform shape to <lat_len * lon_len, time dim, 1>
        data_list = np.reshape(grid_map_list, (len(grid_map_list), -1, 1) )
        data_list = data_list.transpose( (1, 0, 2))

        # swap central row with the last row, use the central row as target series
        data_list[ [-50, -1] ] = data_list[ [-1, -50] ]
        
        if config.is_scaled:
            data = data_list.tolist()
            for i in range(len(data)):
                scaler = MinMaxScaler( config.feature_range )
                data[i] = scaler.fit_transform(data[i]).tolist()
                data[i].append( [scaler] )
            return np.array(data)
        else:
            return data

    def get_batch_data(self, data):
        config = self.config
        batch_data = []
        for i in range(config.timestep, data.shape[-2]):
            batch_data.append(data[:, i - config.timestep : i])
            if len(batch_data) == config.batch_size:
                # <batch_size, config.n + 1, config.timestep, 1>
                batch_data = np.array(batch_data)
                yield batch_data[:, : -1, :, :], batch_data[:, -1, : -1, :], batch_data[:, -1, -1, :]

                batch_data = []
    def divide_three_ds(self, dataset):
        config = self.config
        train_end_index = int(dataset.shape[1] / 10 * 9)
        valid_end_index = int(dataset.shape[1] / 10 * 0)
        #train_end_index = -2016
        #valid_end_index = 0

        if config.is_scaled:
            train_end_index -= 1
            valid_end_index -= 1           
            # remove the last scaler element
            return dataset[ :,  : train_end_index], dataset[:,  train_end_index - config.timestep : valid_end_index], dataset[:, valid_end_index - config.timestep : -1]
        else:
            return dataset[ :,  : train_end_index], dataset[:,  train_end_index - config.timestep : valid_end_index], dataset[:, valid_end_index - config.timestep]
