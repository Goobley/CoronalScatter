import enum
import numpy as np
import matplotlib.pyplot as plt
import os

datafiles = [f for f in sorted(os.listdir()) if f.startswith('Output') and f.endswith('.npz')]
data = [np.load(f) for f in datafiles]
# times = [float(d.split('_')[1][:-4]) for d in datafiles]
times = [d['time'].item() for d in data]


for i, d in enumerate(data):
    print(i)
    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    r = d['r']
    ax[0].scatter(r[:, 0], r[:, 1])
    ax[0].set_title('x-y')
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)

    ax[1].scatter(r[:, 2], r[:, 0])
    ax[1].set_title('z-x')
    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)

    fig.suptitle(f'{times[i]:.2f} s')
    # plt.show()
plt.show()