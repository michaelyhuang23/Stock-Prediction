import torch
from torch import nn

class RCell_complex(nn.Module):
    def __init__(self, input_size, inter_size, recurrent_size, hidden = None):
        '''
        inter_size is meant to be between input_size and recurrent_size
        '''
        super(RCell_complex, self).__init__()
        self.hiddenPrep = nn.Linear(recurrent_size, recurrent_size)
        self.currentPrep = nn.Linear(input_size, inter_size)
        self.meshup = nn.Linear(recurrent_size+inter_size, recurrent_size)
        self.hiddenPost = nn.Linear(recurrent_size, recurrent_size)
        self.tanh = nn.Tanh()
        self.hidden = hidden
        self.recurrent_size = recurrent_size
    
    def forward(self, input):
        if self.hidden is None:
            self.hidden = torch.zeros((input.shape[0],self.recurrent_size))
        self.hidden = self.tanh(self.hiddenPost(self.tanh(self.meshup(self.tanh(torch.cat([self.hiddenPrep(self.hidden),self.currentPrep(input)],dim=-1))))))
        return self.hidden

class RecurrentAnalyzer(nn.Module):
    def __init__(self, recurrent_size, input_size=1, output_size=1):
        super(RecurrentAnalyzer, self).__init__()
        self.rnn_cell = RCell_complex(input_size, recurrent_size)
        self.regressor1 = nn.Linear(recurrent_size,recurrent_size)
        self.regressor2 = nn.Linear(recurrent_size,output_size)
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.output_size = output_size
        self.recurrent_size = recurrent_size

    def forward_step(self, input):
        '''
        input is of shape (N, input_size)
        output is of shape (N, output_size)
        '''
        output = self.rnn_cell(input)
        output = self.regressor2(self.relu(self.regressor1(output)))
        return output
    
    def forward(self, inputs):
        '''
        input is of shape (T, N, input_size)
        output should be of shape (T, N, output_size)
        '''
        #T = input.shape[0]
        outputs = []
        for input in inputs:
            output = self.rnn_cell(input)
            output = self.regressor2(self.relu(self.regressor1(output)))[None,...]
            outputs.append(output)
        return torch.cat(outputs,dim=0)

    def init_hidden(self):
        self.rnn_cell.hidden = None