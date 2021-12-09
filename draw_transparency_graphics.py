import csv  # библиотека для чтения/записи в файлы с расширением .csv

import matplotlib.pyplot as plt  # библиотека для построения графиков (в данном случае двумерных)
import numpy as np  # библиотека работающая с векторами

from utils import get_min_max_points  # импорт функции для получения точек минимумов и максимумов

energy = np.arange(0, 2.501, 0.001)  # создаем вектор энергии


def draw_graphic(
        d: np.ndarray,
        e: np.ndarray,
        max_points: list,
        min_points: list,
        w: int,
        b1: int,
        b2: int
) -> None:
    """Функция построяения графика проницаемости с отмеченными точками минимумов и максимумов"""
    plt.plot(e, d)  # построение графика проницаемости, зависящего от энергии
    # отмечаем точки максимумов
    plt.plot(
        e[max_points],
        d[max_points],
        'x',
        color='red',
        label=f'E_1 = {round(e[max_points[0]], 4)}'
    )
    # отмечаем точки минимумов
    plt.plot(
        e[min_points],
        d[min_points],
        'x',
        color='green'
    )
    plt.grid()  # включение сетки
    plt.legend()  # отображение подписей на графике
    plt.title(f'Проницаемость ДБКС (w={w}нм, b1={b1}нм, b2={b2}нм)')   # подпись графика (названия)
    plt.xlabel('E (eV)')  # подпись оси абсцисс
    plt.ylabel('D')   # подпись оси ординат
    plt.savefig(f'pictures/d/{w}_{b1}_{b2}.jpg')  # сохранение графи
    plt.show()   # отображение графика


if __name__ == '__main__':
    # контекстный менеджер для открытия и автоматического закрытия файла по выходу из вложенности
    with open('data_sets/params_d.csv', 'r') as file:
        # инициализация класса считывающего данные в файл .csv
        reader = csv.reader(file)
        # цикл проходящий по всем строкам файла
        for row in reader:
            w = int(float(row[0]))  # получения значения ширины ямы
            b1 = int(float(row[1]))  # получения значения ширины первого барьера
            b2 = int(float(row[2]))  # получения значения ширины второго барьера
            d = np.array(row[3:], dtype='float')  # получения вектора значениями проницаемости структуры
            # получение точек минимумов и максимумов
            min_points, max_points = get_min_max_points(d)
            # построение графика проницаемости, зависящего от энергии
            draw_graphic(d, energy[:len(d)], max_points, min_points, w, b1, b2)

