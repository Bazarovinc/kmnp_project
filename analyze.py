from get_integral_difference_and_draw_graphics import get_integral_difference
import csv
from utils import normalize_data
import numpy as np

if __name__ == '__main__':
    with open('data_sets/normalized_current.csv', 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    s_cube, w_cube, b1_cube, b2_cube = get_integral_difference(
        rows,
        'U^3',
        lambda v: v ** 3
    )
    print(f'For U^3 S={s_cube[0]} w={w_cube[0]}, b1={b1_cube[0]}, b2={b2_cube[0]}')
    s_square, w_square, b1_square, b2_square = get_integral_difference(
        rows,
        'U^2',
        lambda v: v ** 2
    )
    print(f'For U^2 S={s_square[0]} w={w_square[0]}, b1={b1_square[0]}, b2={b2_square[0]}')
    s_sqrt, w_sqrt, b1_sqrt, b2_sqrt = get_integral_difference(
        rows,
        '√U',
        lambda v: np.sqrt(v)
    )
    print(f'For √U S={s_sqrt[0]} w={w_sqrt[0]}, b1={b1_sqrt[0]}, b2={b2_sqrt[0]}')
    s_exp, w_exp, b1_exp, b2_exp = get_integral_difference(
        rows,
        'e^U',
        lambda v: normalize_data(np.exp(v))
    )
    print(f'For e^U S={s_exp[0]} w={w_exp[0]}, b1={b1_exp[0]}, b2={b2_exp[0]}')

