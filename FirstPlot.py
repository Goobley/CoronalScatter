import enum
import numpy as np
import matplotlib.pyplot as plt
import os

datafiles = [f for f in sorted(os.listdir()) if f.startswith('Output') and f.endswith('.txt')]
data = [np.loadtxt(f) for f in datafiles]
times = [float(d.split('_')[1][:-4]) for d in datafiles]


for i, d in enumerate(data):
    print(i)
    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    ax[0].scatter(d[:, 0], d[:, 1])
    ax[0].set_title('x-y')
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)

    ax[1].scatter(d[:, 2], d[:, 0])
    ax[1].set_title('z-x')
    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)

    fig.suptitle(f'{times[i]:.2f} s')
    plt.show()