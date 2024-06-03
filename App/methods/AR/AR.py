
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
import json
import os

from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List

from ..method import BaseMethodMeta
from ..pipeline import Pipeline
from ..dataset import OneByOneDataset


class ARLinear(nn.Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.p = p
        self.ar = nn.Linear(p, 1)
        
    def forward(self, x):
        return self.ar(x)
    

class ARPipeline(Pipeline, metaclass=BaseMethodMeta):
    
    def __init__(self, model_path: str, preprocess_params: Dict, postprocess_params: Dict) -> None:
        super().__init__(preprocess_params, postprocess_params)
        
        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
            model_params = json.load(f)
            
        self.win_size = model_params["win_size"]
        self.horizon = model_params["horizon"]
        self.batch_size = model_params["batch_size"]
        self.model = ARLinear(self.win_size)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
        self.model = self.model.to(self.device)
        
    def detect_core(self, ts_data: pd.DataFrame, time_name: str="timestamp", metric_name: str="metric") -> np.ndarray:
        print (f"timeseries shape {ts_data.shape}")
        
        ts_time = ts_data[time_name].values
        ts_value = ts_data[metric_name].values
        data_loader = DataLoader(
            dataset=OneByOneDataset(ts_time, ts_value, self.win_size, self.horizon),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        with torch.no_grad():
            for _, X_metric, Y, _ in data_loader:
                X_metric = X_metric.to(self.device)
                output = self.model(X_metric)
                scores.append(Y.numpy() - output.numpy())
        scores = np.abs(np.concatenate(scores).squeeze())
        return np.concatenate([np.zeros((self.win_size, )), scores])