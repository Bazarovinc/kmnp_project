import csv  # библиотека для чтения/записи в файлы с расширением .csv

from get_integral_difference_and_draw_graphics import get_integral_difference

if __name__ == '__main__':
    # извлечение нормализованных значений тока
    with open('data_sets/normalized_current.csv', 'r') as file:
        reader = csv.reader(file)
        # сохранения всех строк из файла в массив для множественной обработки этих значений
        rows = [row for row in reader]
    # получения результатов интегральной разности и суммы разностей для зависимости U^3
    d, s = get_integral_difference(
        rows,
        'U^3',
        lambda v: v ** 3
    )
    # вывод параметров структуры и результатов разностей
    print(f'For U^3 {d}, {s}')
    # получения результатов интегральной разности и суммы разностей для зависимости U^2
    d, s = get_integral_difference(
        rows,
        'U^2',
        lambda v: v ** 2
    )
    # вывод параметров структуры и результатов разностей
    print(f'For U^2 {d}, {s}')
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

