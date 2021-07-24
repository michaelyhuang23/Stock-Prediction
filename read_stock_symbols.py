import pandas as pd

data = pd.read_csv('nasdaq.csv')
data = data[['Symbol','Name']]
stocks = [(symbol,name) for name, symbol in zip(data['Name'].values,data['Symbol'].values) if symbol.isalpha()]
data = pd.DataFrame(stocks, columns=['Symbol', 'Name'])
print(data.head())
data.to_csv('stock_names.csv')