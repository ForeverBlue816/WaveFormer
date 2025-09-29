# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import numpy as np


class EarlyStop():
    def __init__(self, patience:float = -1, max_delta:float = 0) -> bool:
        self.patience = patience
        self.max_delta = max_delta
        self.counter = 0
        self.min_val_metric = np.inf
        self.max_val_metric = 0

    def evaluate_decreasing_metric(self, val_metric):
        """
            e.g. loss or rmse
        """
        if self.patience == -1:
            return False
        
        if val_metric <= self.min_val_metric:
            self.min_val_metric = val_metric
            self.counter = 0
        elif val_metric <= (self.min_val_metric + self.max_delta):
            self.counter = 0
        elif val_metric > (self.min_val_metric + self.max_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        
        return False

    def evaluate_increasing_metric(self, val_metric):
        """
            e.g. accuracy or auroc
        """
        if self.patience == -1:
            return False
        
        if val_metric >= self.max_val_metric:
            self.max_val_metric = val_metric
            self.counter = 0
        elif val_metric >= (self.max_val_metric - self.max_delta):
            self.counter = 0
        elif val_metric < (self.max_val_metric - self.max_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        
        return False