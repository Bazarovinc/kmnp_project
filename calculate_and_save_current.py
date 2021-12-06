import csv  # библиотека для чтения/записи в файлы с расширением .csv
# библиотека для полученния данных о времени (нужная для оценки затраченного времени на рассчеты)
from datetime import datetime

import numpy as np  # библиотека работающая с векторами

from create_current import create_current  # импорт функции для рассчета тока


if __name__ == '__main__':
    # получение файлового дескриптора
    f_o_i = open('data_sets/params_current.csv', 'a')
    # инициализация класса записывающего данные в файл .csv
    writer_i = csv.writer(f_o_i)
    i = 0  # счетчик всех циклов
    # сохранение времени начала рассчетов
    start = datetime.now()
    s = 13
    s1 = 17
    # в 3 вложенных циклах проход по всем значениям w, b1, b2 и получение значений тока через структуру
    for w in range(19, 21):
        for b1 in range(s, 21):
            for b2 in range(s1, 21):
                s_i = datetime.now()
                if b1 + 1 == b2 or b2 + 1 == b1 or b1 == b2:
                    continue
                # получение вектора со значениями тока через структуру
                current = create_current(w, b1, b2)
                e_i = datetime.now()
                # вывод затраченного времени
                print(f'Cycle {i} takes {e_i - s_i}, finished at: {datetime.now()}')
                # формирование вектора, содержащего параметры структуры и тока
                row_i = np.concatenate((np.array([w, b1, b2]), current))
                # запись полученных выше значений в файл
                writer_i.writerow(row_i)
                i += 1
            s1 = 10
        s = 10
    # закрытие файла по его файловому дескриптору
    f_o_i.close()
    end = datetime.now()
    # вывод общего затраченного времени на рассчеты
    print(end - start)
