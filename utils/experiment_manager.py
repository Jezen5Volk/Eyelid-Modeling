import torch
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.data_management import Mat2TVT, Preprocessor, Custom_EMG, Jitter, MaskRand
from models.RNN_rect import EMG_RNN
from utils.training_overhead import Trainer

class Experiment:
    def __init__(self):
        return


    def __call__(self, params, data, epochs = 20, patience = 2):
        params = self.form_experiments(params)
        self.epochs = epochs
        self.patience = patience

        iters = len(params['t_win'])
        best_loss = []
        for i in range(iters):
            print(f'************************************************************\nRunning Experiment {i} of {iters}\n************************************************************')
            metrics = self.run_experiment(params, data, epochs, patience, i)
            loss = min(metrics['Validation Loss'])
            best_loss.append(loss)
        idx = np.argmin(best_loss)
        best_params = self.best_params(params, idx)

        return best_params
        

    def form_experiments(self, params):
        #windowing parameters
        t_win = params['t_win']
        t_stride = params['t_stride']
        t_lookahead = params['t_lookahead']

        #data transformation parameters
        p_transform = params['p_transform']
        sigma = params['sigma']
        p_mask = params['p_mask']

        #Learning parameters
        batch_size = params['batch_size']
        lr = params['learning_rate']
        dropout = params['dropout']

        t_win, t_stride, t_lookahead, p_transform, sigma, p_mask, batch_size, lr, dropout = np.meshgrid(t_win, t_stride, t_lookahead, p_transform, sigma, p_mask, batch_size, lr, dropout)

        #windowing parameters
        params['t_win'] = t_win.ravel()
        params['t_stride'] = t_stride.ravel()
        params['t_lookahead'] = t_lookahead.ravel()

        #data transformation parameters
        params['p_transform'] = p_transform.ravel()
        params['sigma'] = sigma.ravel()
        params['p_mask'] = p_mask.ravel()

        #Learning parameters
        params['batch_size'] = batch_size.ravel()
        params['learning_rate'] = lr.ravel()
        params['dropout'] = dropout.ravel()

        return params


    def run_experiment(self, params, data, epochs, patience, i = 0):
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        '''
        Preprocessing
        '''
        #Window and rectify the EMG data
        preprocessor = Preprocessor(params['t_win'][i], params['t_lookahead'][i], params['t_stride'][i])
        X_train_wr, y_train_wr, init_state_train = preprocessor.win_rect(X_train, y_train)
        X_val_wr, y_val_wr, init_state_val = preprocessor.win_rect(X_val, y_val)

        #Load into custom torch.Dataset object, which applies our data augmentation (Jitter, random masking)
        transform = v2.RandomApply(torch.nn.ModuleList([Jitter(params['sigma'][i]), MaskRand(params['p_mask'][i])]), p = params['p_transform'][i])

        train_data = Custom_EMG(X_train_wr, y_train_wr, init_state_train, transform = transform)
        train_dataloader = DataLoader(train_data, int(params['batch_size'][i]), shuffle = True)

        val_data = Custom_EMG(X_val_wr, y_val_wr, init_state_val, transform = None)
        val_dataloader = DataLoader(val_data, int(params['batch_size'][i]), shuffle = False)

        '''
        Training
        '''
        (train_features, _), train_labels = next(iter(train_dataloader))
        model = EMG_RNN(train_features.size(), train_labels.size(), 2, params['dropout'][i])
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = params['learning_rate'][i])

        trainer = Trainer(train_dataloader, val_dataloader, model, loss_fn, optimizer, int(params['batch_size'][i]), epochs, patience)
        metrics = trainer.train()

        return metrics


    def best_params(self, params, idx):
        #windowing parameters
        params['t_win'] = params['t_win'][idx]
        params['t_stride'] = params['t_stride'][idx]
        params['t_lookahead'] = params['t_lookahead'][idx]

        #data transformation parameters
        params['p_transform'] = params['p_transform'][idx]
        params['sigma'] = params['sigma'][idx]
        params['p_mask'] = params['p_mask'][idx]

        #Learning parameters
        params['batch_size'] = params['batch_size'][idx]
        params['learning_rate'] = params['learning_rate'][idx]
        params['dropout'] = params['dropout'][idx]

        return params