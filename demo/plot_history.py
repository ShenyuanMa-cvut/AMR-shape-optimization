import pickle
import sys

from matplotlib import pyplot as plt

import numpy as np

def main(*args):
    if len(args) == 1:
        print('No history data was given')
        return

    for history_data in args[1:]:
        with open(history_data, 'rb') as f:
            history = pickle.load(f)
        #plt.plot(history['oc'])
        plt.plot(history['vol'])
        #plt.plot(np.array(history['compl']))

    plt.show()

if __name__ == '__main__':
    main(*sys.argv)