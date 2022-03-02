import numpy as np
import pandas as pd

def sliding_window_iter(series, size):
    """series is a column of a dataframe"""
    for start_row in range(len(series) - size + 1):
        yield series[start_row:start_row + size]


df = pd.DataFrame({'A': list(range(100, 501, 100)),
                   'B': list(range(-20, -15)),
                   'C': [0, 1, 2, None, 4]},
                 )

A = list(sliding_window_iter(df['C'], 3))
A = 1