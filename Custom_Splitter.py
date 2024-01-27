from sklearn.model_selection import train_test_split
import os

class CustomSplitter:
    def __init__(self, df, valid_pct=0.2, seed=None):
        self.train_idx, self.valid_idx = train_test_split(
            range(len(df)), test_size=valid_pct, random_state=seed
        )

    def __call__(self, _):
        # The splitter function expects a callable, so we implement the __call__ method.
        return (list(self.train_idx), list(self.valid_idx))
