import csv
from datetime import datetime

import numpy as np

from create_d import create_d

if __name__ == '__main__':
    f_o_d = open('data_sets/params_d.csv', 'w')
    writer_d = csv.writer(f_o_d)
    i = 0
    start = datetime.now()
    flag = True
    for w in range(10, 21):
        for b1 in range(10, 21):
            for b2 in range(10, 21):
                s_i = datetime.now()
                d = create_d(w, b1, b2)
                e_i = datetime.now()
                print(f'Cycle {i} takes {e_i - s_i}, finished at: {datetime.now()}')
                row_d = np.concatenate((np.array([w, b1, b2]), d))
                writer_d.writerow(row_d)
                i += 1
    f_o_d.close()
    end = datetime.now()
    print(end - start)
