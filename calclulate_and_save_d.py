import csv
from datetime import datetime

import numpy as np

from create_d import create_d

if __name__ == '__main__':
    f_o_d = open('data_sets/params_d.csv', 'a')
    writer_d = csv.writer(f_o_d)
    i = 0
    start = datetime.now()
    flag = True
    for w in range(10, 21):
        for b in range(10, 21):
            s_i = datetime.now()
            b1, b2 = b, b
            d = create_d(w, b1, b2)
            e_i = datetime.now()
            print(f'Cycle {i}.0 takes {e_i - s_i}, finished at: {datetime.now()}')
            row_d = np.concatenate((np.array([w, b1, b2]), d))
            writer_d.writerow(row_d)
            s_i = datetime.now()
            b1, b2 = b, b + 1
            d = create_d(w, b1, b2)
            e_i = datetime.now()
            print(f'Cycle {i}.1 takes {e_i - s_i}, finished at: {datetime.now()}')
            row_d = np.concatenate((np.array([w, b1, b2]), d))
            writer_d.writerow(row_d)
            s_i = datetime.now()
            b1, b2 = b2, b1
            d = create_d(w, b1, b2)
            e_i = datetime.now()
            print(f'Cycle {i}.2 takes {e_i - s_i}, finished at: {datetime.now()}')
            row_d = np.concatenate((np.array([w, b1, b2]), d))
            writer_d.writerow(row_d)
            i += 1
    f_o_d.close()
    end = datetime.now()
    print(end - start)
