import csv
from datetime import datetime

import numpy as np

from create_I import create_I

VV = np.arange(0, 2.001, 0.001)
E = np.arange(0, 2.001, 0.001)

if __name__ == '__main__':
    f_o_i = open('data_sets/params_current.csv', 'a')
    writer_i = csv.writer(f_o_i)
    i = 0
    start = datetime.now()
    s = 12
    for w in range(10, 21):
        for b in range(s, 21):
            s_i = datetime.now()
            b1, b2 = b, b + 1
            ii = create_I(w, b1, b2)
            e_i = datetime.now()
            print(f'Cycle {i}.0 takes {e_i - s_i}, finished at: {datetime.now()}')
            row_i = np.concatenate((np.array([w, b1, b2]), ii))
            writer_i.writerow(row_i)
            s_i = datetime.now()
            b1, b2 = b2, b1
            ii = create_I(w, b1, b2)
            e_i = datetime.now()
            print(f'Cycle {i}.1 takes {e_i - s_i}, finished at: {datetime.now()}')
            row_i = np.concatenate((np.array([w, b1, b2]), ii))
            writer_i.writerow(row_i)
            i += 1
        s = 10
    f_o_i.close()
    end = datetime.now()
    print(end - start)
