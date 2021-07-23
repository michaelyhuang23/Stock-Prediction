import pandas as pd

data = pd.read_csv('nasdaq.csv')
data = data[['Symbol','Name']]
symbols = [symbol for symbol in data['Symbol'].values if (not '/' in symbol) and (not '^' in symbol) and (not '\'' in symbol)]
symbols = pd.DataFrame(symbols,columns=['Symbol'])
data.update(symbols)
print(data.head())
data.to_csv('stock_names.csv')