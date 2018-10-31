from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn

class DARNN():
    def __init__(self, config, scope = 'DA_RNN'):
        # placeholder
        self.input_keep_prob = tf.placeholder(tf.float32)
        self.output_keep_prob = tf.placeholder(tf.float32)

        self.X = tf.placeholder(tf.float32, shape = [None, config.n, config.timestep, config.input_size])
        self.Y_prev = tf.placeholder(tf.float32, shape = [None, config.timestep - 1, 1])
        self.Y = tf.placeholder(tf.float32, shape = [None, 1])
        
        self.config = config

        # <batch_size, n , timestep * input_size>
        x_flat = tf.reshape(self.X, shape = [-1, config.n, config.timestep * config.input_size])
        # <batch_size, timestep, n * input_size>
        x_t = tf.reshape(tf.transpose(self.X, perm = [0, 2, 1, 3]), shape = [-1, config.timestep, config.n * config.input_size])
        
        with tf.variable_scope(scope + '_Encoder'):

            encoder_lstms = [tf.contrib.rnn.BasicLSTMCell(hid_size, forget_bias = 0.0) for hid_size in config.hidden_sizes]
            if config.input_keep_prob < 1.0 or config.output_keep_prob < 1.0:
                encoder_lstms = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.input_keep_prob,
                                                                  output_keep_prob = self.output_keep_prob) for lstm in encoder_lstms]
            encoder_cells = tf.contrib.rnn.MultiRNNCell(encoder_lstms, state_is_tuple = True)

            # batch_size * last_hidden_size
            encoder_s_state = encoder_cells.zero_state(config.batch_size, tf.float32)
            
            last_hidden_size = config.hidden_sizes[-1]

            # timestep * <batch_size, last_hidden_size>
            X_encodeds = []

            # attention layer
            ''' !!! w, e are reused '''
##            en_w_e = tf.Variable(tf.truncated_normal(shape = [2 * last_hidden_size, config.timestep]), dtype = tf.float32)
##            en_u_e = tf.Variable(tf.truncated_normal(shape = [config.timestep * config.input_size, config.timestep]), dtype = tf.float32)
##            en_v_e = tf.Variable(tf.truncated_normal(shape = [config.timestep, 1]))

            # LSTM run by step
            for t in range(config.timestep):
                
                en_w_e = tf.Variable(tf.truncated_normal(shape = [2 * last_hidden_size, config.timestep]), dtype = tf.float32)
                en_u_e = tf.Variable(tf.truncated_normal(shape = [config.timestep * config.input_size, config.timestep]), dtype = tf.float32)
                en_v_e = tf.Variable(tf.truncated_normal(shape = [config.timestep, 1]))
                
                # batch_size * 2m
                h_s_concat = tf.concat([encoder_s_state[-1].h, encoder_s_state[-1].c], axis = 1)

                hs_part = tf.matmul(h_s_concat, en_w_e)
                

                # n * <batch_size, 1>
                e_ks = []
                for i in range(config.n):
                    e_k = hs_part + tf.matmul(x_flat[:, i, :], en_u_e)
                    e_k = tf.matmul( tf.nn.tanh(e_k), en_v_e)

                    e_ks.append(e_k)

                e_ks = tf.transpose(e_ks, perm = [1, 0, 2])
                e_ks = tf.reshape(e_ks, shape = [-1, config.n])
                # <batch_size, n>
                alpha_k = tf.nn.softmax(e_ks)
                #print('a_k_diag', tf.matrix_diag(alpha_k))

                # <batch_size, 1, n> = <batch_size, 1, n> matmul <batch_size, n, n>
                
                # x_tilde = tf.matmul( tf.reshape(tf.matmul(x_t[:, t, :], en_feature_e), shape = (-1, 1, config.n)) , tf.matrix_diag(alpha_k))
                # one feature
                x_tilde = tf.matmul( tf.reshape(x_t[:, t, :], shape = (-1, 1, config.n)) , tf.matrix_diag(alpha_k))
                # <batch_size, n>
                x_tilde = tf.reshape(x_tilde, shape = [-1, config.n])

                
                #print('State size:', encoder_cells.state_size)
                encoder_h_state, encoder_s_state = encoder_cells.call(x_tilde, encoder_s_state)

                X_encodeds.append(encoder_h_state)
    
        with tf.variable_scope(scope + '_Decoder'):
            decoder_lstms = [tf.contrib.rnn.BasicLSTMCell(hid_size) for hid_size in config.hidden_sizes]
            if config.input_keep_prob < 1.0 or config.output_keep_prob < 1.0:
                decoder_lstms = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.input_keep_prob,
                                                                  output_keep_prob = self.output_keep_prob) for lstm in decoder_lstms]
            decoder_cells = tf.contrib.rnn.MultiRNNCell(decoder_lstms, state_is_tuple = True)

            decoder_s_state = decoder_cells.zero_state(config.batch_size, tf.float32)
            
            last_hidden_size = config.hidden_sizes[-1]

            # attention layer
            ''' !!! w, e are reused '''
##            de_w_e = tf.Variable(tf.truncated_normal(shape = [2 * last_hidden_size, last_hidden_size]), dtype = tf.float32)
##            de_u_e = tf.Variable(tf.truncated_normal(shape = [last_hidden_size, last_hidden_size]), dtype = tf.float32)
##            de_v_e = tf.Variable(tf.truncated_normal(shape = [last_hidden_size, 1]))

            de_w_tilde = tf.Variable(tf.truncated_normal(shape = [last_hidden_size + 1, 1]), dtype = tf.float32)
            de_b_tilde = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32))

            # lstm run by step
            for t in range(config.timestep):
                
                de_w_e = tf.Variable(tf.truncated_normal(shape = [2 * last_hidden_size, last_hidden_size]), dtype = tf.float32)
                de_u_e = tf.Variable(tf.truncated_normal(shape = [last_hidden_size, last_hidden_size]), dtype = tf.float32)
                de_v_e = tf.Variable(tf.truncated_normal(shape = [last_hidden_size, 1]))

                
                # batch_size * 2m
                d_s_concat = tf.concat([decoder_s_state[-1].h, decoder_s_state[-1].c], axis = 1)
                ds_part = tf.matmul(d_s_concat, de_w_e)

                # timestep * <batch_size, 1>
                l_ks = []
                for i in range(config.timestep):
                    l_k = ds_part + tf.matmul(X_encodeds[t], de_u_e)
                    l_k = tf.matmul( tf.nn.tanh(l_k), de_v_e)

                    l_ks.append(l_k)

                l_ks = tf.transpose(l_ks, perm = [1, 0, 2])
                l_ks = tf.reshape(l_ks, shape = [-1, config.timestep])
                # <batch_size, timestep>
                beta_k = tf.nn.softmax(l_ks)
                
                # <batch_size, m>
                c_t = None
                for i in range(config.timestep):
                    if c_t is None:
                        c_t = X_encodeds[i] * tf.reshape(beta_k[:, i], shape = [-1, 1])
                    else:
                        c_t += X_encodeds[i] * tf.reshape(beta_k[:, i], shape = [-1, 1])

                # decoder input
                # concat y_t and c_t, shape is (batch_size, m + 1)
                y_tilde = tf.concat([ tf.reshape(self.Y_prev[:, max(0, t - 1)], shape = [-1, 1]), c_t], axis = 1)
                y_tilde = tf.matmul(y_tilde, de_w_tilde) + de_b_tilde

                # run lstm cell
                de_h_state, decoder_s_state = decoder_cells.call(y_tilde, decoder_s_state)

            infer_w_y = tf.Variable(tf.truncated_normal(shape = [last_hidden_size + last_hidden_size, last_hidden_size]), dtype = tf.float32)
            infer_b_w = tf.Variable(tf.constant(0.0, shape = [last_hidden_size], dtype = tf.float32))

            infer_v_y = tf.Variable(tf.truncated_normal(shape = [last_hidden_size, 1]), dtype = tf.float32)
            infer_b_v = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32))

            # compute predicted value
            infer_concat = tf.concat([decoder_s_state[-1].h, X_encodeds[-1]], axis = 1)
            y_pred = tf.matmul(infer_concat, infer_w_y) + infer_b_w
            y_pred = tf.matmul(y_pred, infer_v_y) + infer_b_v

        # assignment
        self.loss = tf.losses.mean_squared_error(self.Y, y_pred)
        self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss)
        
        self.Y_pred = y_pred

    def train(self, batch_data, y_scaler, sess):
        all_loss = []
        all_perc_loss = []
        all_rmse = []
        for ds in batch_data:
            _, ls, pred = sess.run([self.train_op, self.loss, self.Y_pred], feed_dict = {self.X : ds[0], self.Y_prev : ds[1], self.Y : ds[2],
                                                              self.input_keep_prob : self.config.input_keep_prob,
                                                              self.output_keep_prob : self.config.output_keep_prob})
            
            y_pre_list = []
            y_real_list = []
            for j in range(len(ds[2])):
                if self.config.is_scaled:
                    y_pre_list.append(y_scaler.inverse_transform([ pred[j] ]))
                    y_real_list.append(y_scaler.inverse_transform([ ds[2][j] ]))

            loss = np.mean( np.divide(abs(np.subtract(y_pre_list, y_real_list)), y_real_list))
            rmse = np.sqrt(np.mean(np.subtract(y_pre_list, y_real_list) ** 2))
        
            all_perc_loss.append(loss)
            all_rmse.append(rmse)
            all_loss.append(ls)
        return np.mean(all_loss), np.mean(all_perc_loss), np.mean(all_rmse)

    def predict(self, batch_data, y_scaler, sess):

        all_loss = []
        all_perc_loss = []
        all_rmse = []
        for ds in batch_data:
            ls, pred = sess.run([self.loss, self.Y_pred], feed_dict = {self.X : ds[0], self.Y_prev : ds[1], self.Y : ds[2],
                                                        self.input_keep_prob : 1.0,
                                                        self.output_keep_prob : 1.0})
            if self.config.is_scaled:
                y_pre_list = []
                y_real_list = []
                for j in range(len(ds[2])):
                    y_pre_list.append(y_scaler.inverse_transform([ pred[j] ]))
                    y_real_list.append(y_scaler.inverse_transform([ ds[2][j] ]))

                loss = np.mean( np.divide(abs(np.subtract(y_pre_list, y_real_list)), y_real_list))
                rmse = np.sqrt(np.mean(np.subtract(y_pre_list, y_real_list) ** 2))
            
                all_perc_loss.append(loss)
                all_rmse.append(rmse)
            all_loss.append(ls)

        return np.mean(all_loss), np.mean(all_perc_loss), np.mean(all_rmse)
        


    
            

  
