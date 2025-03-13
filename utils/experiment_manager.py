import torch
import optuna
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.data_management import Mat2TVT, Preprocessor, Custom_EMG, Jitter, MaskRand
from models.RNN_rect import EMG_RNN
from utils.training_overhead import Trainer

class Experiment:
    def __init__(self):
        return

    def __call__(self, param_choices, data, model, n_trials = 100, epochs = 50):
        study = optuna.create_study(direction='minimize')
        study.optimize(Optunamize(param_choices, data, model, epochs), n_trials, n_jobs = -1)
        trial = study.best_trial
        
        return trial.params
        

    def run_experiment(self, params, data, model, trial = None, epochs = None, patience = 5):
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        model = model() #call wrapper object to instantiate fresh model object every time run_experiment is called

        '''
        Preprocessing
        '''
        #Window and rectify the EMG data
        preprocessor = Preprocessor(params['t_win'], params['t_lookahead'], params['t_stride'])
        X_train_wr, y_train_wr, init_state_train = preprocessor.win_rect(X_train, y_train)
        X_val_wr, y_val_wr, init_state_val = preprocessor.win_rect(X_val, y_val)

        #Load into custom torch.Dataset object, which applies our data augmentation (Jitter, random masking)
        transform = v2.RandomApply(torch.nn.ModuleList([Jitter(params['sigma']), MaskRand(params['p_mask'])]), p = params['p_transform'])

        train_data = Custom_EMG(X_train_wr, y_train_wr, init_state_train, transform = transform)
        train_dataloader = DataLoader(train_data, int(params['batch_size']), shuffle = True)

        val_data = Custom_EMG(X_val_wr, y_val_wr, init_state_val, transform = None)
        val_dataloader = DataLoader(val_data, int(params['batch_size']), shuffle = False)

        '''
        Training
        '''
        (train_features, _), train_labels = next(iter(train_dataloader))
        model.initialize(train_features.size(), train_labels.size(), params)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
        
    
        trainer = Trainer(train_dataloader, val_dataloader, model, loss_fn, optimizer, int(params['batch_size']), trial, epochs, patience)
        metrics = trainer.train()

        return metrics
    

    def optuna_interface(self, trial, param_choices, data, model, epochs):
        params = {}
        #Windowing Parameters
        params['t_win'] = trial.suggest_categorical('t_win', param_choices['t_win'])
        params['t_stride'] = trial.suggest_categorical('t_stride', param_choices['t_stride'])
        params['t_lookahead'] = trial.suggest_categorical('t_lookahead', param_choices['t_lookahead'])

        #Data Transformation Parameters
        params['p_transform'] = trial.suggest_categorical('p_transform', param_choices['p_transform'])
        params['sigma'] = trial.suggest_categorical('sigma', param_choices['sigma'])
        params['p_mask'] = trial.suggest_categorical('p_mask', param_choices['p_mask'])

        #Learning Parameters
        params['batch_size'] = trial.suggest_categorical('batch_size', param_choices['batch_size'])
        params['learning_rate'] = trial.suggest_categorical('learning_rate', param_choices['learning_rate'])
        params['dropout'] = trial.suggest_categorical('dropout', param_choices['dropout'])
        
        metrics = self.run_experiment(params, data, model, trial, epochs)

        return min(metrics['Validation Loss'])
    

    def test_model(self, best_params, data, model, weights):
        '''
        Preprocessing
        '''
        X_test, y_test = data["X_test"], data["y_test"]
        preprocessor = Preprocessor(best_params['t_win'], best_params['t_lookahead'], best_params['t_stride'])
        X_test_wr, y_test_wr, init_state_test = preprocessor.win_rect(X_test, y_test)
        test_data = Custom_EMG(X_test_wr, y_test_wr, init_state_test, transform = None)
        test_dataloader = DataLoader(test_data, int(best_params['batch_size']), shuffle = False)

        '''
        Testing
        '''
        loss_fn = torch.nn.MSELoss()
        (test_features, _), test_labels = next(iter(test_dataloader))
        model.initialize(test_features.size(), test_labels.size(), best_params)
        model.load_state_dict(weights)
        tester = Trainer(None, None, model, loss_fn, None, None, None) #None parameters are relevant for model training
        test_metrics, preds = tester.test_model(test_dataloader)

        return test_metrics, preds
        

            
    

class Optunamize:
    def __init__(self, param_choices, data, model, epochs):
        self.param_choices = param_choices
        self.data = data
        self.model = model
        self.epochs = epochs
    
    def __call__(self, trial):
        experiment = Experiment()
        val_loss = experiment.optuna_interface(trial, self.param_choices, self.data, self.model, self.epochs)
        return val_loss