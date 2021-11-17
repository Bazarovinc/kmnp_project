from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from constants import voltage
from utils import normalize_data
from csv import reader


def draw_graphics(
        current: np.ndarray,
        v: np.ndarray,
        cube_i: np.ndarray,
        w: int,
        b1: int,
        b2: int,
        s: float,
        depend_type: str
) -> None:
    plt.plot(v, current)
    plt.plot(v, cube_i, label=depend_type)
    plt.title(f'ВАХ ДБКС с (w={w}, b1={b1}, b2={b2})нм (s={round(s, 4)})')
    plt.xlabel('U')
    plt.ylabel('I')
    plt.legend()
    plt.grid()
    plt.savefig(f'pictures/diff/{depend_type}/{w}_{b1}_{b2}.jpg')
    plt.show()


def simpson(y: np.ndarray, n: int) -> float:
    h = (y[-1] - y[0]) / n
    y_0 = y[0]
    y_n = y[-1]
    s = 0
    for i in range(1, n - 1):
        if i % 2 != 0:
            c = 1
        else:
            c = -1
        s += y[i] * (3 + c)
    return h / 3 * (y_0 + y_n + s)


def draw_3d(
        p1: np.ndarray,
        p2: np.ndarray,
        s: np.ndarray,
        depend_type: str,
        x_name: str,
        y_name: str
) -> None:
    fig = go.Figure(data=go.Scatter3d(
        x=p1,
        y=p2,
        z=s,
        marker=dict(
            size=10,
            color=s,
            colorscale='plasma',
            showscale=True,
        ),
        line=dict(
            color='darkblue',
            width=5
        )
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_name + ', нм',
            yaxis_title=y_name + ', нм',
            zaxis_title='S',
    ),
        title=f'Зависимость интегрального расхождения от {x_name} и {y_name} {depend_type}'
    )
    fig.show()
    fig = go.Figure(data=go.Scatter3d(
        x=p1,
        y=p2,
        z=s,
        mode='markers',
        marker=dict(
            size=10,
            color=s,
            colorscale='plasma',
            showscale=True,

        ),
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_name + ', нм',
            yaxis_title=y_name + ', нм',
            zaxis_title='S',
        ),
        title=f'Зависимость интегрального расхождения от {x_name} и {y_name} {depend_type}'
    )
    fig.show()


def sort_lists(
        s: dict,
        w: List[float],
        b1: List[float],
        b2: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_s = []
    new_b1 = []
    new_w = []
    new_b2 = []
    s = dict(sorted(s.items(), key=lambda item: item[1]))
    for k, v in s.items():
        new_s.append(v)
        new_w.append(w[k])
        new_b1.append(b1[k])
        new_b2.append(b2[k])
    return np.array(new_s), np.array(new_w), np.array(new_b1), np.array(new_b2)


def get_integral_difference(
        rows: list,
        depend_type: str,
        depend_func
):
    w_list = []
    b1_list = []
    b2_list = []
    s_dict = {}
    i = 0
    for row in rows:
        w = int(float(row[0]))
        b1 = int(float(row[1]))
        b2 = int(float(row[2]))
        w_list.append(w)
        b1_list.append(b1)
        b2_list.append(b2)
        current = np.array(row[3:], dtype='float')
        n_voltage = normalize_data(voltage[:len(current)])
        n_current = depend_func(n_voltage)
        s_dict[i] = np.abs(simpson(current, len(current)) - simpson(n_current, len(n_current)))
        draw_graphics(current, n_voltage, n_current, w, b1, b2, s_dict[i], depend_type)
        i += 1
    s_sorted, w_sorted, b1_sorted, b2_sorted = sort_lists(s_dict, w_list, b1_list, b2_list)
    draw_3d(w_sorted, b1_sorted, s_sorted, depend_type, 'w', 'b1')
    draw_3d(w_sorted, b2_sorted, s_sorted, depend_type, 'w', 'b2')
    draw_3d(b1_sorted, b2_sorted, s_sorted, depend_type, 'b1', 'b2')
    return s_sorted, w_sorted, b1_sorted, b2_sorted
