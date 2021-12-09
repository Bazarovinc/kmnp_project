from datetime import datetime

from matplotlib import pyplot as plt

from constants import *
from create_current import create_current
from utils import get_min_max_points, get_15_point, find_end_point, normalize_data


if __name__ == '__main__':
    w = int(input('Введите w (ширину квантовой ямы): '))
    b1 = int(input('Введите b1 (ширину первого барьера): '))
    b2 = int(input('Введите b2 (ширину второго барьера): '))
    s = datetime.now()
    current = create_current(w, b1, b2)  # получение тока при заданных параметрах структуры w, b1, b2
    e = datetime.now()
    min_points, max_points = get_min_max_points(current)  # получение всех точек минимумов и максимумов
    max_point = max_points[0]
    min_point = min_points[1]
    print(e - s)
    plt.plot(voltage[:len(current)], current)  # построение графика ВАХ
    # отмечаем точки минимумов на графике ВАХ
    plt.plot(voltage[min_points], current[min_points], 'x', color='green', label='maxs')
    # отмечаем точки максимумов на графике ВАХ
    plt.plot(voltage[max_points], current[max_points], 'x', color='red', label='mins')
    plt.grid()  # включение сетки
    plt.title(f'Изначальный график ВАХ (w={w}, b1={b1}, b2={b2})нм')
    plt.xlabel('U')  # подпись оси абсцисс
    plt.ylabel('I')  # подпись оси ординат
    plt.show()  # отображение графика
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
    point_15 = int(np.where(current == get_15_point(current[:max_point + 1], current[max_point]))[0])
    plt.plot(cur_voltage, current)
    plt.plot(cur_voltage[max_point], current[max_point], 'x', color='red', label='Точка максимума')
    plt.plot(cur_voltage[point_15], current[point_15], 'x', color='green', label='15% от пика')
    plt.title(f'ВАХ (w={w}, b1={b1}, b2={b2})нм')
    plt.grid()  # включение сетки
    plt.show()  # отображение графика
    plt.legend()
    plt.xlabel('U')  # подпись оси абсцисс
    plt.ylabel('I')  # подпись оси ординат
    plt.show()
    normalized_current = normalize_data(current[0:point_15])
    normalized_voltage = normalize_data(cur_voltage[0:point_15])
    plt.plot(normalized_voltage, normalized_current)
    plt.grid()  # включение сетки
    plt.show()  # отображение графика
    plt.legend()
    plt.title(f'Нормализованный отрезок ВАХ (в 15% от пика) (w={w}, b1={b1}, b2={b2})нм')
    plt.xlabel('U')  # подпись оси абсцисс
    plt.ylabel('I')  # подпись оси ординат
    plt.show()
