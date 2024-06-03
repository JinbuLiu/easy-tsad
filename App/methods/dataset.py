
import torch
import torch.utils.data
import numpy as np

class OneByOneDataset(torch.utils.data.Dataset):
    
    def __init__(self, ts_time: np.ndarray, ts_metric: np.ndarray, window_size: int, horizon: int):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        
        self.len = ts_time.shape[0]
        self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
        
        X = torch.zeros((self.sample_num, self.window_size))    # window of time
        Y = torch.zeros((self.sample_num, self.window_size))    # window of metric
        Z = torch.zeros((self.sample_num, self.horizon))
        M = torch.zeros((self.sample_num, self.window_size))    # window of nan
        
        for i in range(self.sample_num):
            X[i, :] = torch.from_numpy(ts_time[i : i + self.window_size])
            Y[i, :] = torch.from_numpy(ts_metric[i : i + self.window_size])
            Z[i, :] = torch.from_numpy(ts_metric[i + self.window_size : i + self.window_size + self.horizon])
            M[i, np.argwhere(np.isnan(ts_metric[i : i + self.window_size]))] = 1
            
        self.time_samples = X
        self.metric_samples = Y
        self.horizon_samples = Z
        self.mask_samples = M
        
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return [self.time_samples[index, :], self.metric_samples[index, :], self.horizon_samples[index, :], self.mask_samples[index, :]]
    