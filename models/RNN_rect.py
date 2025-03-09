import torch

class EMG_RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        


