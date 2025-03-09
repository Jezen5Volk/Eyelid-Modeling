import torch

class EMG_RNN(torch.nn.Module):

    def __init__(self, input_size, H1, N1, dropout):
        super().__init__()
        self.BN1 = torch.nn.BatchNorm1d(input_size)
        self.RNN1 = torch.nn.RNN(input_size, H1, num_layers = N1, batch_first = True, nonlinearity = 'relu', dropout = dropout)
        
    def forward(self, X):
        X = self.BN1(X)
        X = self.RNN1(X)
        return X
