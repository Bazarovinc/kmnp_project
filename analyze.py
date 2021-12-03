from get_integral_difference_and_draw_graphics import get_integral_difference
import csv
from utils import normalize_data
import numpy as np

if __name__ == '__main__':
    with open('data_sets/normalized_current.csv', 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
        d, s = get_integral_difference(
        rows,
        'U^3',
        lambda v: v ** 3
    )
    print(f'For U^3 {d}, {s}')
    d, s = get_integral_difference(
        rows,
        'U^2',
        lambda v: v ** 2
    )
    print(f'For U^2 {d}, {s}')
    d, s = get_integral_difference(
        rows,
        '√U',
        lambda v: np.sqrt(v)
    )
    print(f'For √U {d}, {s}')
    d, s = get_integral_difference(
        rows,
        'e^U',
        lambda v: normalize_data(np.exp(v))
    )
    print(f'For e^U {d}, {s}')

