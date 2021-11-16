from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import signal


def normalize_data(data: np.ndarray) -> np.ndarray:
    return (data - min(data)) / (max(data) - min(data))


def get_min_max_points(vector: np.ndarray) -> Tuple[List[int], List[int]]:
    df = pd.DataFrame(vector, columns=['data'])
    df['min'] = df.iloc[signal.argrelextrema(
        df.data.values,
        np.less_equal,
        order=5
    )[0]]['data']
    df['max'] = df.iloc[signal.argrelextrema(
        df.data.values,
        np.greater_equal,
        order=5
    )[0]]['data']
    mins = df[df['min'].notnull() == True].index.to_list()
    maxs = df[df['max'].notnull() == True].index.to_list()
    return mins, maxs
