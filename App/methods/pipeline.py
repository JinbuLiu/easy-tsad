
import sys
sys.path.extend(["..", "../.."])

import numpy as np
import pandas as pd

from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor(object):
    
    def __init__(
        self, 
        normalization: str, 
        diff_order: int, 
        fill_na: bool, 
        missing_na: bool,
        win_size: int = 3,
        **kwargs
    ):
        self.normalization = normalization
        self.diff_order = diff_order
        self.missing_na = missing_na         # 是否补充缺失的值，缺失的值设置为 NaN
        self.fill_na = fill_na               # 是否填充缺失的值，缺失的值通过插值的方式配置
        self.win_size = win_size             # 缺失值填充的窗口大小
        
    def __call__(self, ts_data: pd.DataFrame, time_name: str="timestamp", metric_name: str="metric", start_date: pd.Timestamp=None, end_date: pd.Timestamp=None, freq: str=None):
        ts_data = self.__fill_na(ts_data, time_name, metric_name, freq, start_date, end_date)
        ts_data = self.__normalize(ts_data, metric_name)
        return self.__diff(ts_data, metric_name)
        
    def __diff(self, ts_data: pd.DataFrame, metric_name: str):
        values: np.ndarray = ts_data[metric_name].values
        for _ in range(self.diff_order):
            values = np.pad(values, ((0,1)), 'edge') - np.pad(values, ((1,0)), 'edge')
            values = values[:-1]
        ts_data[metric_name] = values
        return ts_data
        
    def __normalize(self, ts_data: pd.DataFrame, metric_name: str):
        if not self.normalization or self.normalization == "raw":
            return ts_data
        elif self.normalization == "min_max":
            na_index = ts_data[metric_name].isnull()
            ts_data.loc[~na_index, [metric_name]] = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts_data.loc[~na_index, [metric_name]])
            return ts_data
        elif self.normalization == "z_score":
            na_index = ts_data[metric_name].isnull()
            ts_data.loc[~na_index, [metric_name]] = StandardScaler().fit_transform(ts_data.loc[~na_index, [metric_name]])
            return ts_data
        else:
            raise ValueError("normalization must be one of raw, min_max, z_score") 
        
    def __fill_na(self, ts_data: pd.DataFrame, time_name: str, metric_name: str, freq: str, start_date: pd.Timestamp=None, end_date: pd.Timestamp=None):
        ts_data[time_name] = pd.to_datetime(ts_data[time_name], unit="s")
        ts_data = ts_data.sort_values(by=[time_name], ascending=[True])  
        
        if not start_date:
            start_date = ts_data[time_name].iloc[0]
        if not end_date:
            end_date = ts_data[time_name].iloc[-1]
    
        if self.fill_na:
            metric_median_name = f"{metric_name}_median_"
            ts_data[metric_median_name] = ts_data[metric_name].rolling(window=self.win_size, center=False).median()
            time_index = self.__get_date_range(start_date, end_date, freq, time_name)
            time_index[time_name] = pd.to_datetime(time_index[time_name])
            
            ts_data[time_name] = pd.to_datetime(ts_data[time_name])
            ts_data = pd.merge(time_index, ts_data.copy(), on=[time_name], how='left')
            ts_data[metric_median_name] = ts_data[metric_median_name].ffill()
            ts_data[metric_median_name] = ts_data[metric_median_name].bfill()
            
            ts_data.loc[ts_data[metric_name].isnull(), metric_name] = ts_data[metric_median_name]
            ts_data = ts_data.drop(labels=metric_median_name, axis=1)
            ts_data[metric_name] = ts_data[metric_name].fillna(0)
            
        elif self.missing_na:
            time_index = self.__get_date_range(start_date, end_date, freq, time_name)
            time_index[time_name] = pd.to_datetime(time_index[time_name])
            
            ts_data[time_name] = pd.to_datetime(ts_data[time_name])
            ts_data = pd.merge(time_index, ts_data.copy(), on=[time_name], how='left')
        
        ts_data[time_name] = [int(dt.timestamp()) for dt in ts_data[time_name]]
        return ts_data
                
    def __get_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str, time_name: str):
        time_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        time_index = pd.DataFrame(time_index, columns=[time_name])
        return time_index
    

class Postprocessor(object):
    
    def __init__(self, **kwargs):
        pass
    

class Pipeline(object):
    
    def __init__(self, preprocess_params: Dict, postprocess_params: Dict) -> None:
        self.preprocessor = Preprocessor(**preprocess_params)
        self.postprocessor = Postprocessor(**postprocess_params)
        
    def detect(self, ts_data: pd.DataFrame, freq: str, time_name: str="timestamp", metric_name: str="metric", start_date: pd.Timestamp=None, end_date: pd.Timestamp=None) -> np.ndarray:
        ts_data = self.preprocessor(ts_data, time_name, metric_name, start_date, end_date, freq)
        ts_result = self.detect_core(ts_data, time_name, metric_name)
        
    def detect_core(self, ts_data: pd.DataFrame, time_name: str="timestamp", metric_name: str="metric"):
        pass