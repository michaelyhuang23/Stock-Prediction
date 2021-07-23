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

    def load_stock(self, file_path: str = 'stock_names.csv'):
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
        self.stock_data = np.load(dir_path+'/stock_data.npy')
        self.symbol_set = set(self.stock_symbols)
    
    def load_stocks(self, symbols : List):
        data = yf.download(symbols, period='1y', interval='1d',threads=False)
        self.stock_data = data['Open'].to_numpy()
        print(self.stock_data.shape)
                
database = StockDatabase()
#database.read_data()
database.load_stock()
database.save_data()

