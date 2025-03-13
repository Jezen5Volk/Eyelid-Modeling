import torch

class EMG_RNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def initialize(self, shape_in, shape_out, params):
        '''
        Input shape: (N, L, W, C)
        where N is batch size, L is sequence length, W is num of windows, C is num of channels

        Output shape: (N, L, 3, W, C)
        where N is batch size, L is sequence length, W is num of windows, C is num of channels

        Note that sequence length will always differ and number of channels may differ between i/o
        The 3 in the output shape refers to cartesian XYZ coordinate dimensions
        '''
        
        #EMG Layers
        self.BN1 = torch.nn.BatchNorm1d(shape_in[-1])
        self.RNN1 = torch.nn.RNN(input_size = shape_in[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNN2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')
        self.aff1 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])
        self.aff2 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])
        self.aff3 = torch.nn.Linear(in_features = shape_in[1], out_features = shape_out[1])

        #Kinematic Layers
        self.BN2 = torch.nn.BatchNorm2d(shape_out[-1])
        self.RNNX1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNY1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNZ1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNX2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')
        self.RNNY2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')
        self.RNNZ2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')

        #property attributes
        self._shape_out = shape_out
        
    def forward(self, X, pred):
        '''
        EMG Estimator
        '''
        #we expect incoming X to be of shape (N, L, C)
        X = X.permute(0, 2, 1) #BN expects X to be of shape (N, C, L)
        X = self.BN1(X) 
        
        X = X.permute(0, 2, 1) #RNN expects X to be of shape (N, L, C)
        X, _  = self.RNN1(X) 
        X, _ = self.RNN2(X)

        H = X.permute(0, 2, 1) #affine expects X to be of shape (N, C, L)
        X = self.aff1(H)
        Y = self.aff2(H)
        Z = self.aff3(H)

        H = torch.stack((X, Y, Z), dim = 2) #shape is now (N, C, 3, L)
        H = H.permute(0, 3, 2, 1) #now we return the expected shape of (N, L, 3, C)

        '''
        Kinematic Estimator
        '''
        #we expect incoming predictions to be of shape (N, L, 3, C)
        pred = pred.permute(0,3,2,1) #BN2d expectes pred to be of shape (N, C, H, W)
        pred = self.BN2(pred)

        #RNN layers expect pred to be of shape (N, L, C)
        pred = pred.permute(0,3,2,1) 
        X_pred, _ = self.RNNX1(pred[:, :, 0, :])
        X_pred, _ = self.RNNX2(X_pred)
        Y_pred, _ = self.RNNY1(pred[:, :, 1, :])
        Y_pred, _ = self.RNNY2(Y_pred)
        Z_pred, _ = self.RNNY1(pred[:, :, 2, :])
        Z_pred, _ = self.RNNZ2(Z_pred)
        P = torch.stack((X_pred, Y_pred, Z_pred), dim = 2)

        return H + P
    
    @property
    def shape_out(self):
        return self._shape_out
    


class EMG_RNN_Wrapper():
    def __init__(self):
        return
    def __call__(self):
        model = EMG_RNN()
        return model