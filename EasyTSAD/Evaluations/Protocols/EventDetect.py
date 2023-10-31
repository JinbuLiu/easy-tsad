from dataclasses import dataclass
from typing import Type

import numpy as np
from .. import MetricInterface, EvalInterface

@dataclass
class Precision(MetricInterface):
    name: str
    num: int = 1
    detected: float = 0.0
    
    def add(self, other_metric):
        self.num += 1
        self.detected += other_metric.detected
    
    def avg(self):
        if self.num != 0:
            self.detected = float(self.detected) / self.num
            
    def to_dict(self):
        return {
            self.name: {
                "detected": self.detected
            }
        }
        
        
class EventDetect(EvalInterface):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Event Detected"
        
    def calc(self, scores, labels, margins) -> type[MetricInterface]:
        idx = np.argmax(scores)
        detected = 0
        if labels[idx] == 1:
            detected = 1
        return Precision(
            self.name,
            detected=float(detected)
        )
    