import scipy.io
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class Mat2TVT:

    def __init__(self, eye_bool, electrode_list, marker_list, filepath):
        self.eye_bool = eye_bool
        self.electrode_list = electrode_list
        self.marker_list = marker_list
        self.X = None
        self.y = None
        self.filepath = filepath
        self.identifier = None

        

    def TVT_split(self, train, val, test, seed = True):
        if seed:
            np.random.seed(42) # for consistent train/val/test splits


        #Calculate useful numbers
        num_trials = self.identifier.shape[0]
        num_train = int(train*num_trials)
        num_val = int(val*num_trials)


        #All possible trial indices
        available_idx = np.arange(num_trials)


        #Make sure interpolated labels do not affect ability to verify model performance
        idx = np.where(self.identifier[:,2] == 'True')[0]
        X_train = self.X[:, idx, :]
        y_train = self.y[:, :, idx, :]
        id_train = self.identifier[idx, :2]


        #Remove indices we should no longer access, then form indices
        available_idx = np.delete(available_idx, idx)
        np.random.shuffle(available_idx)
        train_idx = available_idx[:num_train-len(idx)]
        val_idx = available_idx[num_train-len(idx):num_train - len(idx) + num_val]
        test_idx = available_idx[num_train - len(idx) + num_val:]


        #combine interpolated training data with the rest of the training data
        X_train = np.hstack((X_train, self.X[:, train_idx, :]))
        y_train = np.dstack((y_train, self.y[:, :, train_idx, :]))
        id_train = np.vstack((id_train, self.identifier[train_idx, :2]))
        id_train = id_train[:,0]


        #Create validation data
        X_val = self.X[:, val_idx, :]
        y_val = self.y[:, :, val_idx, :]
        id_val = self.identifier[val_idx, 0]


        #Create test data
        X_test = self.X[:, test_idx, :]
        y_test = self.y[:, :, test_idx, :]
        id_test = self.identifier[test_idx, 0]

        return {"X_train": X_train, "y_train": y_train, "id_train": id_train, "X_val": X_val, "y_val": y_val, "id_val": id_val, "X_test": X_test, "y_test": y_test, "id_test": id_test}


    def load_data(self):
        mat = scipy.io.loadmat(self.filepath, simplify_cells = True)

        #Collect Data
        data = np.zeros((4564, 1, 5))
        for blink_key in mat['ForSamantha']['emg_with_notchfilter']:
            for sub_key in mat['ForSamantha']['emg_with_notchfilter'][blink_key]:
                trial = []
                for electrode_key in self.electrode_list:
                    trial.append(mat['ForSamantha']['emg_with_notchfilter'][blink_key][sub_key][electrode_key])
                trial = np.dstack(trial) #reshape to (4564, T, 5) where T is number of trials
                data = np.hstack((data, trial)) #stack all trials along 1st axis --> (4564, N, 5)


        #Collect Label/Identifier
        identifier = []
        label = np.zeros((300, 3, 1, 5))
        for blink_key in mat['ForSamantha']['kinem']:
            for sub_key in mat['ForSamantha']['kinem'][blink_key]:
                trial = []
                for marker_key in self.marker_list:
                    trial.append(mat['ForSamantha']['kinem'][blink_key][sub_key][marker_key])
                trial = np.stack(trial, axis = -1) #reshape to (300, 3, T, 7) where T is number of trials

                #Identifier markers for trials with incomplete data
                throwaway = False
                if trial.shape[0] != 300:
                    throwaway = True

                #If not discarding a trial, stack it's label
                if not(throwaway):
                    label = np.dstack((label, trial)) #stack all trials along 2nd axis --> (300, 3, N, 7)
                
                #Generate ID for each trial
                for num in range(trial.shape[2]):
                    id_string = sub_key + '_' + blink_key + '#' + str(num + 1) 
                    id_eye = self.eye_bool[int(sub_key.split('b')[1]) - 1]
                    identifier.append((id_string, id_eye, False, throwaway))


        #Remove placeholder
        X = data[:, 1:, :] 
        y = label[:, :, 1:, :]
        identifier = np.asarray(identifier)


        #Remove throwaway trials from data/identifier
        idx = np.where(identifier[:,3] == 'True')
        X = np.delete(X, idx, axis = 1)
        identifier = np.delete(identifier, idx, axis = 0)
        identifier = identifier[:, :3]


        #separate NaN indices 
        _, _, n_trial, _ = np.where(np.isnan(y))
        removal = []
        for trial in set(n_trial):
            if np.any(np.isnan(y[299, :, trial, :])) or np.any(np.isnan(y[0, :, trial, :])):
                removal.append(trial)
            else:
                y[:, :, trial, :] = self.cubic_interp(y[:, :, trial, :])
                identifier[trial, 2] = True


        #Remove un-interpolatable trials from data
        X = np.delete(X, removal, axis = 1)
        y = np.delete(y, removal, axis = 2)
        identifier = np.delete(identifier, removal, axis = 0)


        #Flipping Z-Label for Right Eye (Ensuring Coordinate System Uniformity)
        for r_eye_idx in np.where(identifier[:,1] == 'True')[0]:
            y[:, 2, r_eye_idx, :] = -1*y[:, 2, r_eye_idx, :]

        
        #Assign to class in case of TVT split 
        self.X = X
        self.y = y
        self.identifier = identifier
    
        return X, y, identifier
    

    def cubic_interp(self, trial):
        interp_trial = np.empty(np.shape(trial))
        for i in range(trial.shape[-2]):
            for j in range(trial.shape[-1]):
                df = pd.Series(trial[:, i, j]).astype(float)
                interp_trial[:, i, j] = np.asarray(df.interpolate(method = 'cubic', limit_direction='both'))
        return interp_trial

    
    def DMVC_norm(self):
        self.X = self.X - np.mean(self.X, axis = 0) #meansub all EMG data to ensure mu = 0
        self.y = self.y - np.mean(self.y, axis = 0) #meansub all labels to ensure model learns relative motions rather than absolute motions

        #obtain string of subjects
        id = self.identifier[:,0]
        sub = []
        for string in id:
            sub.append(string.split('_')[0])
        sub = np.asarray(sub)

        
        for s_num in set(sub):
            #for each subject, set EMG variance to 1 along the channel axis
            idx = np.where(sub == str(s_num))[0]
            s_data = self.X[:, idx, :]
            self.X[:, idx, :] = s_data/np.std(s_data, axis = (0,1))

            #for each subject, set kinematic variance to 1 along the channel axis
            s_data = self.y[:, :, idx, :]
            self.y[:, :, idx, :] = s_data/np.std(s_data, axis = (0,1,2))
        
        return self.X, self.y
    


class Preprocessor:
    def __init__(self, t_win, t_lookahead, t_stride, Xr = 6103.5, yr = 400):
        self.t_win = t_win
        self.t_lookahead = t_lookahead
        self.t_stride = t_stride
        self.Xr = Xr
        self.yr = yr

    
    def window(self, X, y):
        '''
        Incoming EMG data X of shape (t_in, N, C_in) sampled at a rate of Xr Hz, 
        corresponding kinematic label y of shape (t_out, 3, N, C_out) sampled at a rate of yr Hz

        Given the length of the window in milliseconds, 
        the amount of time ahead to make predictions in milliseconds, 
        and the amount of time to move the window in milliseconds

        Note that t_lookahead must be larger than or equal to t_stride

        return for each trial in X and y (N_1, N_2, ...) a series of windows with the desired characterstics

        ie: 
        X = (4564, N, 5) @ 6103.5 Hz
        y = (300, 3, N, 5) @ 400 Hz

        t_win = 20 ms
        t_lookahead = 50 ms
        t_stride = 10 ms

        Then we would expect to return 

        X_win = (20 ms*Xr, N, N_win, 5)
        y_win = (10 ms*yr, 3, N, N_win, 5)
        init_state = (10ms*yr, 3, N, 5)

        where the temporal offset between any (X,y) pair of windows is equal to t_lookahead:
        X_win[:, :, 0, :] --> window encompassing 0 ms to 20 ms (absolute time)
        y_win[:, :, :, 0, :] --> window encompassing 50 ms to 70 ms (absolute time)

        X_win[:, :, 1, :] --> window encompassing 10 ms to 30 ms (absolute time)
        y_win[:, :, :, 1, :] --> window encompassing 70 ms to 90 ms (absolute time)

        init_state --> window encompassing labels from 30 ms to 50 ms (absolute time)

        
        Truncation of partial windows should avoid the biases that padding might introduce.
        '''
        t_x, N, C_x = np.shape(X)
        t_y, d, N, C_y = np.shape(y)
        t_win = self.t_win*1e-3
        t_lookahead = self.t_lookahead*1e-3
        t_stride = self.t_stride*1e-3
        yr = self.yr
        Xr = self.Xr

        num_win = int((t_y - t_lookahead*yr)//int(t_stride*yr))
    
        X_win = np.empty((int(t_win*Xr), N, num_win, C_x))
        y_win = np.empty((int(t_stride*yr), d, N, num_win, C_y))
        init_state = np.empty((int(t_stride*yr), d, N, C_y))
        for i in range(N):
            init_start = int(t_lookahead*yr) - int(t_stride*yr)
            init_stop = init_start + int(t_stride*yr)
            init_state[:, :, i, :] = y[init_start:init_stop, :, i, :]
            for j in range(num_win):
                x_strt = int(j*t_stride*Xr)
                x_stop = x_strt + int(t_win*Xr)
                X_win[:, i, j, :] = X[x_strt:x_stop, i, :]

                y_strt = int(j*t_stride*yr + t_lookahead*yr)
                y_stop = y_strt + int(t_stride*yr)
                y_win[:, :, i, j, :] = y[y_strt:y_stop, :, i, :]


        return X_win, y_win, init_state


    def rectify(self, X):
        return np.abs(X)


    def freq_spec(self, X): 
        fft = np.fft.fft(X, axis = 0)
        fft_amp = np.sqrt(fft.real**2 + fft.imag**2)
        fft_bins = np.fft.fftfreq(X.shape[0], d = 1/self.Xr)
        return np.log(fft_amp), fft_bins
    

    #Just rectify the EMG data
    def win_rect(self, X, y):
        X_w, y_wr, init_state = self.window(X, y)
        X_wr = self.rectify(X_w)
        return X_wr, y_wr, init_state
    

    #Just return the FFT of the EMG data
    def win_fft(self, X, y):
        X_w, y_wf, init_state = self.window(X, y)
        X_wf, _ = self.freq_spec(X_w)
        return X_wf, y_wf, init_state



class Custom_EMG(Dataset):
    def __init__(self, X, y, init_state, transform = None, target_transform = None):
        if torch.cuda.is_available(): 
            self.X = torch.Tensor(X).to(torch.float32).cuda()
            self.y = torch.Tensor(y).to(torch.float32).cuda()
            self.init_state = torch.Tensor(init_state).to(torch.float32).cuda()
        else: 
            self.X = torch.Tensor(X).to(torch.float32)
            self.y = torch.Tensor(y).to(torch.float32)
            self.init_state = torch.Tensor(init_state).to(torch.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.y.shape[2]
    
    def __getitem__(self, idx):
        trial = torch.Tensor(self.X[:, idx, :]).to(torch.float32)
        label = torch.Tensor(self.y[:,:,idx, :]).to(torch.float32)
        init = torch.Tensor(self.init_state[:, :, idx, :]).to(torch.float32)
        
        if self.transform:
            trial = self.transform(trial)
        
        if self.target_transform:
            label = self.target_transform(label)

        X = (trial, init)

        return X, label
    
    

class Jitter(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def __call__(self, sample):
        if torch.cuda.is_available(): 
            jitter = torch.Tensor(np.random.normal(loc = 0, scale = self.scale, size = sample.shape)).to(torch.float32).cuda()
        else: 
            jitter = torch.Tensor(np.random.normal(loc = 0, scale = self.scale, size = sample.shape)).to(torch.float32).cuda()
        
        return sample + jitter
    

class MaskRand(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def __call__(self, sample):
        if self.p == 0:
            return sample
        elif int(self.p*sample.shape[0]) == 0:
            return sample
        else: 
            idx = np.random.randint(int((1-self.p)*sample.shape[0]))
            sample[idx:idx + int(self.p*sample.shape[0]), :] = torch.zeros(1)

        return sample