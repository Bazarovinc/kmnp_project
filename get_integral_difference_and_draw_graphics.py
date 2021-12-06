from typing import List, Tuple  # библиотека подсказок типов

import numpy as np
import plotly.graph_objects as go  # библиотека для построения графиков (в данном случае трехмерных)
from matplotlib import pyplot as plt  # библиотека для построения графиков (в данном случае двумерных)
from pydantic import BaseModel  # библиотека для создания моделей

from constants import voltage  # импорт напряжения из константных значений
from utils import normalize_data  # импорт функции нормализации


class DiffModel(BaseModel):
    diff: float
    diff_type: str
    w: int
    b1: int
    b2: int


def draw_graphics(
        current: np.ndarray,
        v: np.ndarray,
        need_current: np.ndarray,
        w: int,
        b1: int,
        b2: int,
        depend_type: str,
) -> None:
    """Функция построения двух графиков ВАХ: желаемого и полученного"""
    plt.plot(v, current, 'x-')  # построение полученного графиика ВАХ
    plt.plot(v, need_current, 'x-', color='red', label=depend_type)  # построение желаемого графиика ВАХ
    plt.title(f'ВАХ ДБКС с (w={w}, b1={b1}, b2={b2})нм')  # подпись графика (названия)
    plt.xlabel('U')  # подпись оси абцисс
    plt.ylabel('I')  # подпись оси ординат
    plt.legend()  # отображние подписей на графике
    plt.grid()  # включеиние сетки
    plt.savefig(f'pictures/diff/{depend_type}/{w}_{b1}_{b2}.jpg')  # сохранение графика
    plt.show()  # отображение графика


def simpson(y: np.ndarray, n: int) -> float:
    """Функция реализующая метод Симпсона (вычисление интеграла (площади под граффиком))"""
    h = (y[-1] - y[0]) / n  # вычисление шага
    y_0 = y[0]
    y_n = y[-1]
    s = 0
    for i in range(1, n - 1):
        s += y[i] * 4 if i % 2 != 0 else y[i] * 2
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
            zaxis_title='S'
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
    """Функция сортировки списков ям, барьеров и разностей по значениям разностей"""
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


def dif(current_1: np.ndarray, current_2: np.ndarray) -> float:
    """Функция для получения суммы разностей"""
    return np.abs(np.sum(np.abs(current_1 - current_2)))


def get_integral_difference(
        rows: list,
        depend_type: str,
        depend_func
) -> Tuple[DiffModel, DiffModel]:
    w_list = []  # инициализация списка ширин ям
    b1_list = []  # инициализация списка ширин первых барьеров
    b2_list = []  # инициализация списка ширин вторых барьеров
    s_dict = {}  # инициализация словаря интегральных разностей
    diff_dict = {}  # инициализация словаря суммы разностей
    i = 0  # инициализация счетчика
    # цикл по строкам полученным из файла
    for row in rows:
        w = int(float(row[0]))  # получения значения ширины ямы
        b1 = int(float(row[1]))  # получения значения ширины первого барьера
        b2 = int(float(row[2]))  # получения значения ширины второго барьера
        w_list.append(w)  # добавление ширины ямы в массив
        b1_list.append(b1)  # добавление ширины первого барьера в массив
        b2_list.append(b2)  # добавление ширины второго барьера в массив
        current = np.array(row[3:], dtype='float')  # получения вектора значениями проницаемости структуры
        n_voltage = normalize_data(voltage[:len(current)])  # получение нормализованного значения напряжения
        n_current = depend_func(n_voltage)  # получение вектора значений тока для необходимой функции
        # добавление в словарь суммы разностей между желаемыми значениями тока и полученными для структуры
        diff_dict[i] = dif(current, n_current)
        # добавление в словарь интегральной разности между желаемыми значениями тока и полученными для структуры
        s_dict[i] = np.abs(simpson(current, len(current)) - simpson(n_current, len(n_current)))
        # построение графиков желаемой и полученной ВАХ
        draw_graphics(current, n_voltage, n_current, w, b1, b2, depend_type)
        i += 1
    # сортировка списков ширин ям, барьеров, по значениям интегральной разности для дальнейшего вывода
    # характеристик подходящей стрктуры
    s_sorted, w_sorted, b1_sorted, b2_sorted = sort_lists(s_dict, w_list, b1_list, b2_list)
    # запись результатов в модель
    s_diff = DiffModel(
        diff=s_sorted[0],
        diff_type='Интегральная разность',
        w=w_sorted[0],
        b1=b1_sorted[0],
        b2=b2_sorted[0]
    )
    # сортировка списков ширин ям, барьеров, по значениям сумм разностей для дальнейшего вывода
    # характеристик подходящей стрктуры
    diff_sorted, w_sorted, b1_sorted, b2_sorted = sort_lists(diff_dict, w_list, b1_list, b2_list)
    # запись результатов в модель
    diff = DiffModel(
        diff=diff_sorted[0],
        diff_type='Сумма разностей',
        w=w_sorted[0],
        b1=b1_sorted[0],
        b2=b2_sorted[0]
    )
    # new = []
    # b1_l = []
    # for i in range(len(w_sorted)):
    #     if w_sorted[i] == 10:
    #         new.append(i)
    # s_sorted_10 = s_sorted[new]
    # b1_sorted_10 = b1_sorted[new]
    # b2_sorted_10 = b2_sorted[new]
    # draw_3d(b2_sorted_10, b1_sorted_10, s_sorted_10, depend_type, 'b2', 'b1')
    # draw_3d(w_sorted, b1_sorted, s_sorted, depend_type, 'w', 'b1')
    # draw_3d(w_sorted, b2_sorted, s_sorted, depend_type, 'w', 'b2')
    # draw_3d(b1_sorted, b2_sorted, s_sorted, depend_type, 'b1', 'b2')
    return s_diff, diff
