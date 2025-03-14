import torch
import optuna


class Trainer:
    def __init__(self, train_dl, val_dl, model, loss_fn, optimizer, batch_size, trial, epochs = 10, patience = 10, delta = 0):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else: 
            self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.delta = delta
        self.trial = trial

    
    def train(self, verbose = False):
        self.verbose = verbose
        training_loss = []
        training_avgerr = []
        training_maxerr = []
        validation_loss = []
        validation_avgerr = []
        validation_maxerr = []
        early_stopper = EarlyStopper(self.patience, self.delta)
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_avgerr, train_maxerr = self.train_loop()
            val_loss, val_avgerr, val_maxerr = self.val_loop()

            #appending various metrics
            training_loss.append(train_loss)
            training_avgerr.append(train_avgerr)
            training_maxerr.append(train_maxerr)
            validation_loss.append(val_loss)
            validation_avgerr.append(val_avgerr)
            validation_maxerr.append(val_maxerr)

            #early stopping
            early_stopper(val_loss, self.model)
            if early_stopper.early_stop:
                print(f'Stopped early after epoch: {t}')
                break

            #Optuna pruning
            if self.trial is not None: 
                self.trial.report(val_loss, t)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        print("Done!")

        metrics = {
            'Training Loss': torch.Tensor(training_loss).cpu(),
            'Training Avg Marker Error': torch.Tensor(training_avgerr).cpu(),
            'Training Max Marker Error': torch.Tensor(training_maxerr).cpu(),
            'Validation Loss': torch.Tensor(validation_loss).cpu(),
            'Validation Avg Marker Error': torch.Tensor(validation_avgerr).cpu(),
            'Validation Max Marker Error': torch.Tensor(validation_maxerr).cpu(), 
            'Best Weights': early_stopper.best_weights,
            }

        return metrics


    def train_loop(self):
        size = len(self.train_dl.dataset)
        self.model.train()
        
        running_loss = []
        running_merr = []
        err = 0
        for batch, ((X, P), y) in enumerate(self.train_dl):
            #Handle device switching
            if torch.cuda.is_available(): 
                pred = torch.zeros(y.shape).cuda()
            else: 
                pred = torch.zeros(y.shape)
            
            for win in range(X.shape[-2]): 
                pred = pred.clone().detach()
                pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                P = pred[:, :, :, win, :].clone().detach()
            
            #Tracking metrics
            loss = self.loss_fn(pred, y)
            train_acc, train_avg = self.two_norm3D(pred, y)
            err = torch.max(torch.Tensor([train_acc, err]))
            running_loss.append(loss.item())
            running_merr.append(train_avg)

            #backpropagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            self.optimizer.step()

            if batch % 5 == 0 and self.verbose: 
                loss, current = loss.item(), batch * self.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return torch.mean(torch.Tensor(running_loss)), torch.mean(torch.Tensor(running_merr)), err


    def val_loop(self):
        self.model.eval()
        num_batches = len(self.val_dl)

        val_loss = 0
        val_merr = 0
        err = 0

        with torch.no_grad():
            for (X,P), y in self.val_dl:
                #Handle device switching
                if torch.cuda.is_available(): 
                    pred = torch.zeros(y.shape).cuda()
                else: 
                    pred = torch.zeros(y.shape)
                
                for win in range(X.shape[-2]):
                    pred = pred.clone()
                    pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                    P = pred[:, :, :, win, :].clone()

                val_loss += self.loss_fn(pred, y)
                val_acc, val_avg = self.two_norm3D(pred, y)
                val_merr += val_avg
                err = torch.max(torch.Tensor([val_acc, err]))
                
        
        val_loss /= num_batches
        val_merr /= num_batches
        
        print(f"Validation Error: \n Max Marker Error: {(err):>0.1f}%, Avg Marker Error: {(val_merr):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        return val_loss, val_merr, err

    
    def test_model(self, test_dl):
        self.model.eval()
        test_loss = 0
    
        with torch.no_grad():
            preds = ()
            Y = ()
            for (X,P), y in test_dl:
                #Handle device switching
                if torch.cuda.is_available(): 
                    pred = torch.zeros(y.shape).cuda()
                else: 
                    pred = torch.zeros(y.shape)
                
                for win in range(X.shape[-2]):
                    pred = pred.clone()
                    pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                    P = pred[:, :, :, win, :].clone()
                
                preds = preds + tuple(pred.clone())
                Y = Y + tuple(y)

            Y = torch.stack(Y)
            preds = torch.stack(preds)
            test_loss += self.loss_fn(preds, Y)/len(Y)
            max_err, mean_err = self.two_norm3D(pred, y)
            max_err /= len(Y)
            mean_err /= len(Y)
            

        
        
        print(f"Test Error: \n Max Marker Error: {(max_err):>0.1f}%, Avg Marker Error: {(mean_err):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        metrics = {
                    "Test Max Marker Error": torch.Tensor(max_err).cpu(), 
                    "Test Avg Marker Error": torch.Tensor(mean_err).cpu(), 
                    "Test Loss": torch.Tensor(test_loss).cpu()
                   }


        return metrics, pred

    
    def two_norm3D(self, pred, y, eps = 1e-8):
        X_pred = pred[:,:,0,:]
        X = y[:,:,0,:]
        X_diff = X_pred - X

        Y_pred = pred[:,:,1,:]
        Y = y[:,:, 1, :]
        Y_diff = Y_pred - Y

        Z_pred = pred[:,:,2,:]
        Z = y[:,:, 2, :]
        Z_diff = Z_pred - Z

        magnitude = torch.sqrt(X**2 + Y**2 + Z**2)
        diff = torch.sqrt(X_diff**2 + Y_diff**2 + Z_diff**2)

        err = diff/(magnitude + eps)*100
        max_err = torch.max(err)
        mean_err = torch.mean(err)

        return max_err, mean_err



class EarlyStopper:
    def __init__(self, patience = 10, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False


    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict()
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        elif self.counter < self.patience:
            self.counter += 1
        else: 
            self.early_stop = True



