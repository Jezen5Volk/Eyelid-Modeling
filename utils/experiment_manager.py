import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.data_management import Mat2TVT, Preprocessor, Custom_EMG, Jitter, MaskRand
from models.RNN_rect import EMG_RNN
from utils.training_overhead import Trainer

class Experiment:
    def __init__(self, params):
        self.params = params
        self.experiments = self.form_experiments()
    def __call__(self, epochs, patience):
        return
        
    def form_experiments(self):
        #windowing parameters
        t_win = self.params['t_win']
        t_stride = self.params['t_stride']
        t_lookahead = self.params['t_lookahead']

        #data transformation parameters
        p_transform = self.params['p_transform']
        sigma = self.params['sigma']
        p_mask = self.params['p_mask']

        #Learning parameters
        batch_sze = self.params['batch_size']
        lr = self.params['learning_rate']
        dropout = self.params['dropout']
        

