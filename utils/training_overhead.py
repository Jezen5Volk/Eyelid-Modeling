import torch

class Trainer:
    def __init__(self, train_dl, val_dl, model, loss_fn, optimizer, batch_size, epochs):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.batch_size = batch_size
        self.epochs = epochs

    
    def train(self, verbose = False):
        self.verbose = verbose
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()
            self.val_loop()
        print("Done!")

        return self.model.state_dict()


    def train_loop(self):
        size = len(self.train_dl.dataset)
        self.model.train()

        for batch, (X, y) in enumerate(self.train_dl):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            

            #backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            self.optimizer.step()
            

            if batch % 100 == 0 and self.verbose: 
                loss, current = loss.item(), batch * self.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def val_loop(self):
        self.model.eval()
        num_batches = len(self.val_dl)

        test_loss = 0
        err = 0

        with torch.no_grad():
            for X, y in self.val_dl:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y)
                test_acc = torch.max((pred-y)/y)*100 ##REMINDER TO ADD EPISILON FOR NUMERICAL STABILITY
                err = torch.max((test_acc, err))
        
        test_loss /= num_batches
        
        print(f"Validation Error: \n Max Marker Error: {(err):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    def test_model(self, test_dl):
        self.model.eval()
        num_batches = len(test_dl)

        test_loss = 0
        err = 0

        with torch.no_grad():
            for X, y in test_dl:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y)
                test_acc = torch.max((pred-y)/y)*100
                err = torch.max((test_acc, err))
        
        test_loss /= num_batches
        
        print(f"Test Error: \n Max Marker Error: {(err):>0.1f}%, Avg loss: {test_loss:>8f} \n")
