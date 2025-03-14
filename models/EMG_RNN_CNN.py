import torch

class EMG_RNN_CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def initialize(self, shape_in, shape_out, params):
        '''
        Desired Input shape: (N, L, W, C)
        where N is batch size, L is sequence length, W is num of windows, C is num of channels

        Desired Output shape: (N, L, 3, W, C)
        where N is batch size, L is sequence length, W is num of windows, C is num of channels

        Note that sequence length will always differ and number of channels may differ between i/o
        The 3 in the output shape refers to cartesian XYZ coordinate dimensions


        shape_in forward call: (N, L, C)
        shape_out: (N, L, 3, C)
        '''
        self._shape_out = shape_out
        _, L_in, _, C_in = shape_in
        _, L_out, _, _, C_out = shape_out
        padding = int((params['CNN_kernel'] - 1)/2) 
        
        
        self.EMG = torch.nn.ModuleDict({
            'spatial_CNN': torch.nn.ModuleList([
                torch.nn.BatchNorm1d(C_in), 
                torch.nn.Conv1d(C_in, C_out, params['CNN_kernel'], padding = padding),
                torch.nn.MaxPool1d(params['CNN_kernel'], stride = 1, padding = padding),
            ]), 

            'temporal_CNN': torch.nn.ModuleList([
                torch.nn.BatchNorm1d(L_in), 
                torch.nn.Conv1d(L_in, L_out, params['CNN_kernel'], padding = padding),
                torch.nn.MaxPool1d(params['CNN_kernel'], stride = 1, padding = padding),
            ]), 

            'RNN': torch.nn.ModuleList([
                torch.nn.RNN(input_size = C_out, hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout']),
                torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = C_out, num_layers = 1, batch_first = True, nonlinearity = 'relu')
            ]),

            'affine': torch.nn.ModuleList([
                torch.nn.Linear(L_out, L_out), 
                torch.nn.Linear(L_out, L_out),
                torch.nn.Linear(L_out, L_out)
            ])
        })

        #Kinematic Layers
        self.BN2 = torch.nn.BatchNorm2d(shape_out[-1])
        self.RNNX1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNY1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNZ1 = torch.nn.RNN(input_size = shape_out[-1], hidden_size = params['RNN_hdim'], num_layers = params['RNN_depth'], batch_first = True, nonlinearity = 'relu', dropout = params['dropout'])
        self.RNNX2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')
        self.RNNY2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')
        self.RNNZ2 = torch.nn.RNN(input_size = params['RNN_hdim'], hidden_size = shape_out[-1], num_layers = 1, batch_first = True, nonlinearity = 'relu')

        
        
    def forward(self, X, pred):
        N = X.shape[0]

        '''
        EMG Estimator
        '''
        #we expect incoming X to be of shape (N, L, C)
        X = X.permute(0, 2, 1) #spatial_CNN expects X to be of shape (N, C, L)
        for layer in self.EMG['spatial_CNN']:
            X = layer(X)
        
        X = X.permute(0, 2, 1) #temporal_CNN expects X to be of shape (N, L, C)
        for layer in self.EMG['temporal_CNN']:
            X = layer(X)

        for layer in self.EMG['RNN']:
            X, _ = layer(X)

        
        H = torch.zeros((N, self._shape_out[1], self._shape_out[2], self._shape_out[4])).cuda()
        X = X.permute(0, 2, 1) #affine expects X to be of shape (N, C, L)
        i = 0
        for layer in self.EMG['affine']:
            h = layer(X).relu()
            h = X.permute(0, 2, 1)
            H[:, :, i, :] = h
            i += 1
        
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
    


class EMG_RNN_CNN_Wrapper():
    def __init__(self):
        return
    def __call__(self):
        model = EMG_RNN_CNN()
        return model