import yfinance as yf
import pandas as pd
from typing import List
import os.path
import pickle
import numpy as np

class StockDatabase:
    def __init__(self) -> None:
        self.symbol_set = set([])
        self.stock_symbols = []
        self.stock_names = []

    def read_stock(self, file_path: str = 'stock_names.csv'):
        stock_info = pd.read_csv(file_path)
        symbols = []
        names = []
        for symbol,name in zip(list(stock_info['Symbol']),list(stock_info['Name'])):
            if not symbol in self.symbol_set:
                symbols.append(symbol)
                names.append(name)
        del stock_info
        self.symbol_set.update(symbols)
        self.stock_symbols+=symbols
        self.stock_names+=names
        print(len(self.stock_names))
        self.load_stocks(symbols)

    def save_data(self, dir_path: str = 'stock_database'):
        with open(dir_path+'/stock_names.txt', "wb") as fp:   #Pickling
            pickle.dump(self.stock_names, fp)
        with open(dir_path+'/stock_symbols.txt', "wb") as fp:   #Pickling
            pickle.dump(self.stock_symbols, fp)
        np.save(dir_path+'/stock_data.npy', self.stock_data)
    
    def read_data(self, dir_path: str = 'stock_database'):
        with open(dir_path+'/stock_names.txt', "rb") as fp:   #Pickling
            self.stock_names = pickle.load(fp)
        with open(dir_path+'/stock_symbols.txt', "rb") as fp:   #Pickling
            self.stock_symbols = pickle.load(fp)
        self.stock_data = np.load(dir_path+'/stock_data.npy').astype(np.float32)
        self.symbol_set = set(self.stock_symbols)
    
    def load_stocks(self, symbols : List, lim_size : int = 250):
        stock_data = []
        batch_size = 100
        for i in range(len(symbols)//batch_size):
            batch = symbols[i*batch_size : min(len(symbols),(i+1)*batch_size)]
            data = yf.download(batch, period='1y', interval='1d', threads=True)
            stock_data.append(data['Open'].to_numpy()[:lim_size])
            print(f'reading {(i+1)*batch_size}th stock')
        #data = yf.download(symbols, period='1y', interval='1d',threads=False)
        self.stock_data = np.nan_to_num(np.transpose(np.concatenate(stock_data,axis=1)).astype(np.float32))
        print(self.stock_data.shape)
    
    def normalize(self, data):
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        data -= mean
        if std != 0:
            data /=std
        return data

    def to_data(self, length):
        data = []
        for stock in self.stock_data:
            total_len = len(stock)
            for i in range(total_len//length):
                if stock[i*length]>1e-9:
                    data.append(self.normalize(stock[i*length : (i+1)*length])[None,...])
        data = np.concatenate(data,axis=0)
        np.random.shuffle(data)
        return np.transpose(data)

            


                
# database = StockDatabase()
# database.read_data()
# database.stock_data = np.nan_to_num(database.stock_data)
# # print(database.stock_data[:10,-10:])
# database.save_data()

