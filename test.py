from StockDatabase import StockDatabase
from RNNinner import RecurrentAnalyzer
import torch
import matplotlib.pyplot as plt
import numpy as np

database = StockDatabase()
database.read_data()

prices = torch.tensor(database.normalize(database.get_stock_prices('AAPL',length=2000)))
print(prices.shape)
model = RecurrentAnalyzer(100,10).to('cpu')

model.load_state_dict(torch.load('rnn_inner'))

model.init_hidden()

model.eval()
with torch.no_grad():
	preds = list(model(prices[:50,None,None])[:,0])
	for i in range(len(prices)-50):
		preds.append(model.forward_step(preds[-1][None,...])[0])
	print(preds)
	print(prices[1:])
	plt.plot(np.arange(len(prices)-1),prices[1:])
	plt.plot(np.arange(len(preds)), preds)
	plt.show()
