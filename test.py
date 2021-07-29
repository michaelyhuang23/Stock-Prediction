from StockDatabase import StockDatabase
from RNNbaseline import RecurrentAnalyzer
import torch
import matplotlib.pyplot as plt
import numpy as np

database = StockDatabase()
database.read_data()

prices = torch.tensor(database.normalize(database.get_stock_prices('AAPL',length=1000)))
print(prices.shape)
model = RecurrentAnalyzer(100).to('cpu')

model.load_state_dict(torch.load('rnn_baseline'))

model.init_hidden()

model.eval()

with torch.no_grad():
	preds = model(prices[...,None,None])[:,0,0]
	print(preds)
	print(prices[1:])
	plt.plot(np.arange(len(prices)-1),prices[1:])
	plt.plot(np.arange(len(preds)), preds)
	plt.show()
