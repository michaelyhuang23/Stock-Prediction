from RNNbaseline import RecurrentAnalyzer
from torch import nn
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from StockDatabase import StockDatabase

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

database = StockDatabase()
database.read_data()
#print(database.stock_data.shape)
#print(database.stock_data[:10,-10:])
model = RecurrentAnalyzer(50).to(device)
optimizer = Adam(params=model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

EPOCH = 100
batch_size = 64
val_split = 0.2
length = 100
train_data = torch.tensor(database.to_data(length))
#print(train_data[:10,:10])
train_len = int(val_split*train_data.shape[1])
val_data = train_data[-train_len:]
train_data = train_data[:train_len]
train_data = train_data.to(device)
val_data = val_data.to(device)
#print(train_data.dtype)
for epoch in range(EPOCH):
    lossSum = 0
    for batch_i in range(train_len//batch_size):
        model.init_hidden()
        batch_data = train_data[:, batch_i*batch_size : (batch_i+1)*batch_size][...,None]
        #print(batch_data[:-1,...].shape)
        preds = model(batch_data[:-1,...])
        #print(preds)
        loss = loss_fn(preds,batch_data[1:]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossSum+=loss.detach().item()
    lossSum/=(train_len//batch_size)
    print(f'training epoch {epoch}/{EPOCH}; loss: {lossSum}')
    writer.add_scalar('Loss/train', lossSum, epoch)

