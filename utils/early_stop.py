# -*- coding: utf-8 -*-

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, metric):
        # 期望 metric 越大越好（如 Dice/IoU）
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True