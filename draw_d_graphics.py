import csv

import matplotlib.pyplot as plt
import numpy as np

from utils import get_min_max_points

e = np.arange(0, 1.001, 0.001)


def draw_graphic(
        d: np.ndarray,
        e: np.ndarray,
        max_points: list,
        min_points: list,
        w: int,
        b1: int,
        b2: int
) -> None:
    plt.plot(e, d)
    plt.plot(
        e[max_points],
        d[max_points],
        'x',
        color='red',
        label=f'E_1 = {round(e[max_points[0]], 4)}'
    )
    plt.plot(
        e[min_points],
        d[min_points],
        'x',
        color='green'
        # label=f'E_1 = {round(e[max_point], 4)}'
    )
    plt.grid()
    plt.legend()
    plt.title(f'Проницаемость ДБКС (w={w}нм, b1={b1}нм, b2={b2}нм)')
    plt.xlabel('E (eV)')
    plt.ylabel('D')
    plt.savefig(f'pictures/d/{w}_{b1}_{b2}.jpg')
    plt.show()


if __name__ == '__main__':
    with open('data_sets/params_d.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            w = int(float(row[0]))
            b1 = int(float(row[1]))
            b2 = int(float(row[2]))
            d = np.array(row[3:], dtype='float')
            min_points, max_points = get_min_max_points(d)
            draw_graphic(d, e, max_points, min_points, w, b1, b2)

