from typing import List, Tuple  # библиотека подсказок типов

import numpy as np  # библиотека работающая с векторами
import pandas as pd  # библиотека для работы с большими обемами данных (таблицами)
from scipy import signal  # импортирована для получения точек минимумов и максимумов


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Функция нормализации данных в векторе"""
    return (data - min(data)) / (max(data) - min(data))


def get_min_max_points(vector: np.ndarray) -> Tuple[List[int], List[int]]:
    """Функция для получения всех точек минимумов и максимумов на графике"""
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


def find_end_point(min_point: float, max_point: float, array: np.ndarray) -> float:
    """Функция для нахождения конечного значения тока, чтобы в дальнейшем обрезать график.
    Нужная для построения красивого графика ВАХ"""
    half = (max_point - min_point) * 0.6
    point = (np.abs(array - (min_point + half))).argmin()
    return array[point]


def get_15_point(current: np.ndarray, max_point: float) -> float:
    """Функция для нахождения значения тока, лежащей в 15% слева от пикового значения тока"""
    return current[(np.abs(current - (max_point * 0.85))).argmin()]
