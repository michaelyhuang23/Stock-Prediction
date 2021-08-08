from RNNinner import RecurrentAnalyzer
from torch import nn
import torch
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
# serve command: tensorboard --logdir=runs  
from StockDatabase import StockDatabase
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

database = StockDatabase()
database.read_data()
#print(database.stock_data.shape)
#print(database.stock_data[:10,-10:])
model = RecurrentAnalyzer(100,10).to(device)
optimizer = Adam(model.parameters(),lr=0.0001)
loss_fn = nn.MSELoss()

minLoss = 1e9
EPOCH = 200
batch_size = 64
val_split = 0.2
length = 235
train_data = torch.tensor(database.to_data(length))
#print(train_data[:10,:10])
train_len = int((1-val_split)*train_data.shape[1])
print(train_len)
val_data = train_data[:,train_len:]
train_data = train_data[:,:train_len]
train_data = train_data.to(device)[...,None]
val_data = val_data.to(device)[...,None]
#print(train_data.dtype)
for epoch in range(EPOCH):
    lossSum = 0
    for batch_i in range(train_len//batch_size):
        model.init_hidden()
        batch_data = train_data[:, batch_i*batch_size : (batch_i+1)*batch_size]
        #print(batch_data[:-1,...].shape)
        preds = model(batch_data[:-1,...])
        loss = loss_fn(preds,batch_data[1:]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(model.rnn_cell.hiddenDense.weight.grad)
        lossSum+=loss.detach().item()
    lossSum/=(train_len//batch_size)
    print(f'training epoch {epoch}/{EPOCH}; loss: {lossSum}\n')
    writer.add_scalar('Loss/train', lossSum, epoch)
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            #print(val_data.shape)
            model.init_hidden()
            preds = model(val_data[:-1,...])
            loss = loss_fn(preds, val_data[1:])
            if loss < minLoss:
                minLoss = loss
                torch.save(model.state_dict(), 'rnn_inner')
            print(f'evaluating epoch {epoch}/{EPOCH}; loss: {loss}')
            writer.add_scalar('Loss/val', loss, epoch)

