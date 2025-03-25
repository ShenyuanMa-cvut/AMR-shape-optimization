import pickle
from matplotlib import pyplot as plt

import numpy as np

from utils import fit_power_law

def main(*args):
    with open('data/history_uni_696_11136.pkl', 'rb') as f:
        history_uni = pickle.load(f)
    with open('data/history_adaptive_696_21464.pkl','rb') as f:
        history_amr = pickle.load(f)

    plt.plot(history_amr['oc'],label='oc with AMR')
    plt.plot(history_uni['oc'],label='oc with Uni Ref')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()