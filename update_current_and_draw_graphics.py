import csv  # библиотека для чтения/записи в файлы с расширением .csv
from typing import Optional  # библиотека подсказок типов

import matplotlib.pyplot as plt  # библиотека для построения графиков (в данном случае двумерных)
import numpy as np  # библиотека работающая с векторами
# импорт функции для получения точек минимумов и максимумов и функции для нормализации вектора
from utils import get_min_max_points, normalize_data, get_15_point, find_end_point

# инициализация вектора со значениями напряжения
voltage = np.arange(0, 2.501, 0.001)


def draw_graphic(
        v: np.ndarray,
        i: np.ndarray,
        w: Optional[int] = None,
        b1: Optional[int] = None,
        b2: Optional[int] = None,
        max_point: Optional[int] = None,
        point_15: Optional[int] = None,
        flag: bool = False,
) -> None:
    """Функция для построения графика ВАХ"""
    plt.plot(v, i)  # построение графика ВАХ
    # подпись графика (названия)
    if w and b1 and b2 and max_point:
        plt.title(f'ВАХ (w={w}, b1={b1}, b2={b2})нм')
    elif not flag and w and b1 and b2 and not max_point:
        plt.title(f'Изначальный график ВАХ (w={w}, b1={b1}, b2={b2})нм')
    elif flag and w and b1 and b2 and not max_point:
        plt.title(f'Нормализованный отрезок ВАХ (в 15% от пика) (w={w}, b1={b1}, b2={b2})нм')
    # отмечаем пиковое значение тока на графике ВАХ
    if max_point:
        plt.plot(v[max_point], i[max_point], 'x', color='red', label=f'Пик (U={round(v[max_point], 4)})')
    # отмечаем точку в 15% от пика
    if point_15:
        plt.plot(v[point_15], i[point_15], 'x', color='green', label='15% от пика')
    # отображение подписей на графике
    if max_point or point_15:
        plt.legend()
    plt.grid()   # включение сетки
    plt.xlabel('U')  # подпись оси абсцисс
    plt.ylabel('I')  # подпись оси ординат
    # сохранение графика
    if w and b1 and b2 and max_point and point_15:
        plt.savefig(f'pictures/cvc/{w}_{b1}_{b2}.jpg')
    elif not flag and w and b1 and b2 and not max_point:
        plt.savefig(f'pictures/cvc_before/{w}_{b1}_{b2}.jpg')
    elif flag and w and b1 and b2 and not max_point:
        plt.savefig(f'pictures/normalized_cvc/{w}_{b1}_{b2}.jpg')
    plt.show()  # отображение графика


if __name__ == '__main__':
    # инициализация класса записывающего данные в файл .csv
    writer = csv.writer(open('data_sets/normalized_current.csv', 'w'))
    # контекстный менеджер для открытия и автоматического закрытия файла по выходу из вложенности
    with open('data_sets/params_current.csv', 'r') as f:
        # инициализация класса считывающего данные в файл .csv
        reader = csv.reader(f)
        # цикл проходящий по всем строкам файла
        for row in reader:
            w = int(float(row[0]))  # получения значения ширины ямы
            b1 = int(float(row[1]))  # получения значения ширины первого барьера
            b2 = int(float(row[2]))  # получения значения ширины второго барьера
            current = np.array(row[3:], dtype='float')  # получения вектора значениями проницаемости структуры
            # построение изначального графика ВАХ
            draw_graphic(voltage[:len(current)], current, w, b1, b2)
            # получение точек минимумов и максимумов
            min_points, max_points = get_min_max_points(current)
            min_point = min_points[1]  # точка пикового тока
            max_point = max_points[0]  # точка минимума после ОДП
            # получение последнего значения тока, чтобы избавиться от лишних значений тока
            if len(max_points) >= 2:
                end_value = find_end_point(
                    current[min_point],
                    current[max_point],
                    current[min_point:max_points[1]]
                )
                end_point = np.where(current == end_value)[0]  # нахождение индекса с полученным конечным значением
            else:
                end_point = max_point + int((min_point - max_point) / 2)
            # получение векторов тока и напряжения до конечной точки
            if end_point < len(current):
                current = current[:int(end_point) + 1]
                cur_voltage = voltage[:int(end_point) + 1]
            else:
                cur_voltage = voltage
            # нахождение точки, лежащую слева в 15% от точки с пиковым током
            point_15 = int(np.where(current == get_15_point(current[:max_point + 1], current[max_point]))[0])
            # построение обновленного графика ВАХ
            draw_graphic(cur_voltage, current, w, b1, b2, max_point, point_15)
            # нормализация 85% отрезка от пика тока и напряжения
            normalized_current = normalize_data(current[0:point_15])
            normalized_voltage = normalize_data(cur_voltage[0:point_15])
            # построение графика нормализованного ВАХ
            draw_graphic(normalized_voltage, normalized_current, w, b1, b2, flag=True)
            # запись нормализованного вектора тока с параметрами структуры в файл
            writer.writerow(np.concatenate((np.array([w, b1, b2]), normalized_current)))
