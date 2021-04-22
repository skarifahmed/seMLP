import math
import numpy as np
import pandas as pd


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, days, from_end):
        dataframe = pd.read_csv(filename)
        # i_split = int(len(dataframe) * split)
        i_split = 4348
        # self.data_train = dataframe.get(cols).values[1731:i_split]
        # self.data_train = dataframe.get(cols).values[2840:]
        # self.data_train = dataframe.get(cols).values[2739:]
        self.data_train = dataframe.get(cols).values[7654:]
        # self.data_test  = dataframe.get(cols).values[i_split:]
        # self.data_test  = dataframe.get(cols).values[1731:2840]
        self.data_test  = dataframe.get(cols).values[6994:7654]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.days = days 
        self.from_end = from_end

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        seq_len += self.days

        data_windows = []
        for i in range(self.len_test - (seq_len+0)):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        # x = data_windows[:, :-1]
        # y = data_windows[:, -1, [0]]
        x = data_windows[:, :self.from_end]
        y = data_windows[:, -1, [0]]
        return x,y

    
    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - (seq_len+self.days)):
            x, y = self._next_window(i, seq_len, normalise, "train")
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        while 1:
            '''Yield a generator of training data from filename on given list of cols split for train/test'''
            i = 0
            while i < (self.len_train - (seq_len+self.days)):
                x_batch = []
                y_batch = []
                for b in range(batch_size):
                    if i >= (self.len_train - (seq_len+self.days)):
                        # stop-condition for a smaller final batch if data doesn't divide evenly
                        # print ("Shape x last: ", np.array(x_batch).shape)
                        # print ("Shape y last: ", np.array(y_batch).shape)
                        yield np.array(x_batch), np.array(y_batch)
                        i = 0
                    x, y = self._next_window(i, seq_len, normalise, "train")
                    # print ("x in generate: ", x)
                    # print ("y in generate: ", y)
                    x_batch.append(x)
                    y_batch.append(y)
                    i += 1
                    # print ("Shape x: ", np.array(x_batch).shape)
                    # print ("Shape y: ", np.array(y_batch).shape)
                yield np.array(x_batch), np.array(y_batch)

    def generate_test_batch(self, seq_len, batch_size, normalise):
        while 1:
            '''Yield a generator of training data from filename on given list of cols split for train/test'''
            i = 0
            while i < (self.len_test - (seq_len+self.days)):
                x_batch = []
                y_batch = []
                for b in range(batch_size):
                    if i >= (self.len_test - (seq_len+self.days)):
                        # stop-condition for a smaller final batch if data doesn't divide evenly
                        # print ("Shape x last: ", np.array(x_batch).shape)
                        # print ("Shape y last: ", np.array(y_batch).shape)
                        yield np.array(x_batch), np.array(y_batch)
                        i = 0
                    x, y = self._next_window(i, seq_len, normalise, "test")
                    # print ("x in generate: ", x)
                    # print ("y in generate: ", y)
                    x_batch.append(x)
                    y_batch.append(y)
                    i += 1
                    # print ("Shape x: ", np.array(x_batch).shape)
                    # print ("Shape y: ", np.array(y_batch).shape)
                yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise, dataset_type):
        '''Generates the next data window from the given index location i'''
        # i=0
        # print ("i: ", i)
        # print ("Train length: ", self.len_train)
        # normalise = False
        seq_len += self.days
        if dataset_type == "train":
            window = self.data_train[i:i+seq_len]
        else:
            window = self.data_test[i:i+seq_len]
        # print("---------")
        # print ("Window: ", window[:-5])
        # print ("Seq len: ", seq_len)
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        # x = window[:-1]
        x = window[:self.from_end]
        # t+1
        y = window[-1, [0]]       
        #  Added by me to get t+2 observation
        # y = [self.data_train[i+seq_len][0]]
        # print ("x: ", x)
        # print ("x length: ", x.shape)
        # print ("y: ", y)
        # print ("y length: ", y.shape)
        # print ("x: ", x)
        # print ("y: ", y)
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


    def de_normalise_windows(self, window_data, single_window=False):
        '''De-Normalise window with a base value of zero'''
        de_normalised_data = []
        p0 = []
        i=0
        while i < len(self.data_test):
            p0.append(self.data_test[i][0])
            i += 1
        print ("p0 length, Windows length: ", len(p0), len(window_data))

        window_data = [window_data] if single_window else window_data
        for index, window in enumerate(window_data):
            de_normalised_window = []
            de_normalised_val = p0[index]*(window + 1) 
            de_normalised_window.append(de_normalised_val)
            # de_normalised_window = np.array(de_normalised_window).T # reshape and transpose array back into original multidimensional format
            de_normalised_data.append(de_normalised_window)
        return np.array(de_normalised_data)