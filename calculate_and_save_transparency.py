import csv  # библиотека для чтения/записи в файлы с расширением .csv
# библиотека для полученния данных о времени (нужная для оценки затраченного времени на рассчеты)
from datetime import datetime

import numpy as np  # библиотека работающая с векторами

from create_transparency import create_transparency  # импорт функции для рассчета проницаемости

if __name__ == '__main__':
    # получение файлового дескриптора
    f_o_d = open('data_sets/params_d.csv', 'w')
    # инициализация класса записывающего данные в файл .csv
    writer_d = csv.writer(f_o_d)
    i = 0  # счетчик всех циклов
    # сохранение времени начала рассчетов
    start = datetime.now()
    # в 3 вложенных циклах проход по всем значениям w, b1, b2 и получение значений проницаемости структуры
    for w in range(10, 21):
        for b1 in range(10, 21):
            for b2 in range(10, 21):
                s_i = datetime.now()
                # получение вектора со значениями проницаемости структуры
                d = create_transparency(w, b1, b2)
                e_i = datetime.now()
                # вывод времени затраченного на получение для заданной структуры значений проницаемости
                print(f'Cycle {i} takes {e_i - s_i}, finished at: {datetime.now()}')
                # формирование вектора, содержащего параметры структуры и проницаемости
                row_d = np.concatenate((np.array([w, b1, b2]), d))
                # запись полученных выше значений в файл
                writer_d.writerow(row_d)
                i += 1
    f_o_d.close()
    # закрытие файла по его файловому дескриптору
    end = datetime.now()
    # вывод общего затраченного времени на рассчеты
    print(end - start)
