import csv  # библиотека для чтения/записи в файлы с расширением .csv
from exceptions import WrongAnswer

from get_integral_difference_and_draw_graphics import get_integral_difference

if __name__ == '__main__':
    # извлечение нормализованных значений тока
    with open('data_sets/normalized_current.csv', 'r') as file:
        reader = csv.reader(file)
        # сохранения всех строк из файла в массив для множественной обработки этих значений
        rows = [row for row in reader]
    answer = input('Хотите увидеть полный набор 3-ех мерных графиков или только часть? [да/нет] ').lower()
    flag = False
    if answer == 'да':
        flag = True
    elif answer != 'нет':
        raise WrongAnswer
    # получения результатов интегральной разности и суммы разностей для зависимости U^3
    d, s = get_integral_difference(
        rows,
        'U^3',
        lambda v: v ** 3,
        flag
    )
    # вывод параметров структуры и результатов разностей
    print(f'For U^3 {d}, {s}')
    result_file = open('result.txt', 'w')
    result_file.write(f'\nFor U^3 {d}, {s}')
    _ = input('Введите любой символ и нажмите Enter или просто нажмите Enter ')  # пауза для анализа зависимости куба
    # получения результатов интегральной разности и суммы разностей для зависимости U^2
    d, s = get_integral_difference(
        rows,
        'U^2',
        lambda v: v ** 2,
        flag
    )
    # вывод параметров структуры и результатов разностей
    print(f'For U^2 {d}, {s}')
    result_file.write(f'\nFor U^2 {d}, {s}')
    result_file.close()
    # d, s = get_integral_difference(
    #     rows,
    #     '√U',
    #     lambda v: np.sqrt(v)
    # )
    # print(f'For √U {d}, {s}')
    # d, s = get_integral_difference(
    #     rows,
    #     'e^U',
    #     lambda v: normalize_data(np.exp(v))
    # )
    # print(f'For e^U {d}, {s}')

