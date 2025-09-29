
from typing import List


class MeanSquaredError():
    def __init__(self, squared:bool=True) -> None:
        """
        squared: if True returns MSE value, if False return RMSE value

        inputs:
        y_pred (B, V)
        y (B, V)

        output:
        mse (B, V)
        """
        super().__init__()
        self.squared = squared

        self.N = 0
        self.running_squared_sum = []

    def _squared_sum(self, y_pred, y) -> list:
        squared_sum = []
        for i in range(y_pred.shape[-1]):
            squared_sum_value = ((y[..., i] - y_pred[..., i])**2).sum(axis=0)
            squared_sum.append(squared_sum_value.item())
        
        if len(self.running_squared_sum) == 0:
            return squared_sum
        else:
            return [sum(squared_sum_values) for squared_sum_values in zip(squared_sum, self.running_squared_sum)]

    def __call__(self, y_pred, y) -> list:
        self.N += y_pred.shape[0]
        self.running_squared_sum = self._squared_sum(y_pred, y)

        return [(rss / self.N) if self.squared else (rss / self.N)**0.5 for rss in self.running_squared_sum]

    def compute(self) -> list:
        return [(rss / self.N) if self.squared else (rss / self.N)**0.5 for rss in self.running_squared_sum]

    def reset(self) -> None:
        self.N = 0
        self.running_squared_sum = []


def update_best_stats(best_stats:List, test_stat:float, best_stats_len:int, mode="increasing") -> List:
    if mode == "decreasing":
        # e.g. loss
        if max(best_stats) > test_stat:    # if better than worst score
            best_stats.append(test_stat)

        if len(best_stats) > best_stats_len:
            best_stats = sorted(best_stats)
            best_stats.pop()
    else: 
        # e.g. auroc
        if min(best_stats) < test_stat:    # if better than worst score
            best_stats.append(test_stat)

        if len(best_stats) > best_stats_len:
            best_stats = sorted(best_stats, reverse=True)
            best_stats.pop()

    return best_stats