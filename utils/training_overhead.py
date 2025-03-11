import torch

class Trainer:
    def __init__(self, train_dl, val_dl, model, loss_fn, optimizer, batch_size, epochs):
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

    
    def train(self, verbose = False):
        self.verbose = verbose
        training_loss = []
        validation_loss = []
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            training_loss.append(self.train_loop())
            validation_loss.append(self.val_loop())
        print("Done!")

        return torch.Tensor(training_loss).cpu(), torch.Tensor(validation_loss).cpu()


    def train_loop(self):
        size = len(self.train_dl.dataset)
        self.model.train()
        
        running_loss = []
        for batch, (X, y) in enumerate(self.train_dl):
            #Handle device switching
            if torch.cuda.is_available(): 
                pred = torch.zeros(y.shape).cuda()
            else: 
                pred = torch.zeros(y.shape)
            
            P = pred[:, :, :, 0, :]
            for win in range(X.shape[-2]): 
                pred = pred.clone().detach()
                pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                P = pred[:, :, :, win, :].clone().detach()
            loss = self.loss_fn(pred, y)
            running_loss.append(loss.item())
            
            #backpropagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            self.optimizer.step()

            if batch % 5 == 0 and self.verbose: 
                loss, current = loss.item(), batch * self.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return torch.mean(torch.Tensor(running_loss))


    def val_loop(self):
        self.model.eval()
        num_batches = len(self.val_dl)

        val_loss = 0
        err = 0

        with torch.no_grad():
            for X, y in self.val_dl:
                #Handle device switching
                if torch.cuda.is_available(): 
                    pred = torch.zeros(y.shape).cuda()
                else: 
                    pred = torch.zeros(y.shape)
                P = pred[:, :, :, 0, :]
                for win in range(X.shape[-2]):
                    pred = pred.clone()
                    pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                    P = pred[:, :, :, win, :].clone()

                val_loss += self.loss_fn(pred, y)
                val_acc, val_merr = self.two_norm3D(pred, y)
                err = torch.max(torch.Tensor([val_acc, err]))
                
        
        val_loss /= num_batches
        
        print(f"Validation Error: \n Max Marker Error: {(err):>0.1f}%, Avg Marker Error: {(val_merr):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        return val_loss

    
    def test_model(self, test_dl):
        self.model.eval()
        num_batches = len(test_dl)

        test_loss = 0
        err = 0

        with torch.no_grad():
            for X, y in test_dl:
                #Handle device switching
                if torch.cuda.is_available(): 
                    pred = torch.zeros(y.shape).cuda()
                else: 
                    pred = torch.zeros(y.shape)
                P = pred[:, :, :, 0, :]
                for win in range(X.shape[-2]):
                    pred = pred.clone()
                    pred[:, :, :, win, :] = self.model(X[:, :, win, :], P)
                    P = pred[:, :, :, win, :].clone()

                test_loss += self.loss_fn(pred, y)
                test_acc, test_merr = self.two_norm3D(pred, y)
                err = torch.max(torch.Tensor([test_acc, err]))
        
        test_loss /= num_batches
        
        print(f"Test Error: \n Max Marker Error: {(err):>0.1f}%, Avg Marker Error: {(test_merr):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss

    
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




