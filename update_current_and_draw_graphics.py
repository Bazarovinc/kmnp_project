import csv
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from utils import get_min_max_points, normalize_data


def draw_graphic(
        v: np.ndarray,
        i: np.ndarray,
        w: Optional[int] = None,
        b: Optional[int] = None,
        max_point: Optional[int] = None,
        point_15: Optional[int] = None,
        flag: bool = False,
) -> None:
    plt.plot(v, i)
    if w and b and max_point:
        plt.title(f'ВАХ (w={w}нм, b={b}нм)')
    elif not flag and w and b and not max_point:
        plt.title(f'Изначальный график ВАХ w={w}, b={b}')
    elif flag and w and b and not max_point:
        plt.title(f'Нормализованный отрезок ВАХ (в 15% от пика) (w={w}нм, b={b}нм)')
    if max_point:
        plt.plot(v[max_point], i[max_point], 'x', label=f'Пик (U={round(v[max_point], 4)})')
    if point_15:
        plt.plot(v[point_15], i[point_15], 'x', label='15% от пика')
    if max_point or point_15:
        plt.legend()
    plt.grid()
    plt.xlabel('U')
    plt.ylabel('I')
    if w and b and max_point and point_15:
        plt.savefig(f'pictures/cvc/{w}_{b}.jpg')
    elif not flag and w and b and not max_point:
        plt.savefig(f'pictures/cvc_before/{w}_{b}.jpg')
    elif flag and w and b and not max_point:
        plt.savefig(f'pictures/normalized_cvc/{w}_{b}.jpg')
    plt.show()


def find_end_point(min_point: float, max_point: float, array: np.ndarray) -> float:
    half = (max_point - min_point) * 0.75
    point = (np.abs(array - (min_point + half))).argmin()
    return array[point]


if __name__ == '__main__':
    with open('new_params_current.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            voltage = np.arange(0, 2.001, 0.001)
            if row[0] == 'w':
                continue
            w = int(float(row[0]))
            b = int(float(row[1]))
            current = np.array(row[2:], dtype='float')
            end = len(current)
            draw_graphic(voltage[:end], current, w, b)
            min_points, max_points = get_min_max_points(current)
            min_point = min_points[1]
            max_point = max_points[0]
            end_value = find_end_point(
                current[min_point],
                current[max_point],
                current[min_point:]
            )
            end_point = np.where(current == end_value)
            if end_point[0] < len(current):
                current = current[:int(end_point[0]) + 1]
                voltage = voltage[:int(end_point[0]) + 1]
            point_15 = int(round(0.85 * max_point, 0))
            draw_graphic(voltage, current, w, b, max_point, point_15)
            normalized_current = normalize_data(current[0:point_15])
            normalized_voltage = normalize_data(voltage[0:point_15])
            draw_graphic(normalized_voltage, normalized_current, w, b, flag=True)

