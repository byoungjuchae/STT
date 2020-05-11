import torch
import torch.nn as nn






class RNN (nn.Module):

    def __init__(self,input_size,hidden_size,dropout,bidirectional=True):



        self.rnn=nn.LSTM(input_size,hidden_size=hidden_size,dropout=dropout,bidirectional=True)



    def forward(self,x):


        output,hidden=self.rnn(input)

        return output


