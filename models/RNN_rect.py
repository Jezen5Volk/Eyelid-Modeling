import torch

class EMG_RNN(torch.nn.Module):

    def __init__(self, shape_in, shape_out, N1, dropout):
        '''
        Input shape: (N, L, C)
        where N is batch size, L is sequence length, C is num of channels

        Output shape: (N, L, 3, C)
        where N is batch size, L is sequence length, C is num of channels

        Note that sequence length will always differ and number of channels will usually differ between i/o
        The 3 in the output shape refers to cartesian XYZ coordinate dimensions
        '''
        super().__init__()
        self.BN1 = torch.nn.BatchNorm1d(shape_in[2])
        self.RNN1 = torch.nn.RNN(input_size = shape_in[2], hidden_size = shape_out[3], num_layers = N1, batch_first = True, nonlinearity = 'relu', dropout = dropout)
        self.aff1 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])
        self.aff2 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])
        self.aff3 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])
        
    def forward(self, X):
        #we expect incoming X to be of shape (N, L, C)
        X = X.permute(0, 2, 1) #BN expects X to be of shape (N, C, L)
        X = self.BN1(X) 
        
        X = X.permute(0, 2, 1) #RNN expects X to be of shape (N, L, C)
        X, _  = self.RNN1(X) 
        
        H = X.permute(0, 2, 1) #affine expects X to be of shape (N, C, L)
        X = self.aff1(H)
        Y = self.aff2(H)
        Z = self.aff3(H)

        H = torch.stack((X, Y, Z), dim = 2) #shape is now (N, C, 3, L)
        H = H.permute(0, 3, 2, 1) #now we return the expected shape of (N, L, 3, C)

        return H
