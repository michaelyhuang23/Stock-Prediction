import yfinance as yf
import pandas as pd
from typing import List
import os.path
import pickle
import numpy as np
from datetime import datetime, timedelta

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
    
    def load_stocks(self, symbols : List, size : int = 470):
        stock_data = []
        batch_size = 100
        for i in range(len(symbols)//batch_size):
            batch = symbols[i*batch_size : min(len(symbols),(i+1)*batch_size)]
            data = yf.download(batch, period='1mo', interval='15m', threads=True)
            data = data['Open'].to_numpy()[:size]
            print(data.shape)
            if len(data)<size:
                data = np.pad(data,((0,size-len(data)),(0,0)),'constant',constant_values=0)
            print(data.shape)
            stock_data.append(data)
            print(f'reading {(i+1)*batch_size}th stock')
        #data = yf.download(symbols, period='1y', interval='1d',threads=False)
        self.stock_data = np.nan_to_num(np.transpose(np.concatenate(stock_data,axis=1)).astype(np.float32))
        print(self.stock_data.shape)
    
    def normalize(self, data):
        mask = data>0.01
        new_dat = data[mask]
        minv = np.min(new_dat,axis=0)
        std = np.std(new_dat,axis=0)
        # print(minv,std)
        # print(np.min(data,axis=0),np.std(data,axis=0))
        data[mask] -= minv
        if std != 0:
            data /=std
        return data

    def to_data(self, length):
        data = []
        for stock in self.stock_data:
            total_len = len(stock)
            for i in range(total_len//length):
                if stock[i*length]>1e-9:
                    print(np.min(stock[i*length : (i+1)*length]))
                    data.append(self.normalize(stock[i*length : (i+1)*length])[None,...])
        data = np.concatenate(data,axis=0)
        np.random.shuffle(data)
        return np.transpose(data)

    def get_stock_prices(self, symbol : str, length : int):
        '''
        length should be in 15-mins
        '''
        start = datetime.now()-timedelta(minutes=length*15)
        start = start.strftime("%Y-%m-%d")
        print(start)
        data = yf.Ticker(symbol).history(interval='15m',start=start)
        data = data['Open'].to_numpy()
        return data.astype(np.float32)

                
# database = StockDatabase()
# database.read_stock()
# database.save_data()

# database.read_data()
# database.stock_data = np.nan_to_num(database.stock_data)
# # print(database.stock_data[:10,-10:])
# database.save_data()

