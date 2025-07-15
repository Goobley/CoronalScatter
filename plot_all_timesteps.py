import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("build/scatter.nc")

for i in range(ds.time.shape[0]):
    print(i)
    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    ax[0].scatter(ds.rx[i], ds.ry[i])
    ax[0].set_title('x-y')
    ax[0].set_xlim(-7, 7)
    ax[0].set_ylim(-7, 7)

    ax[1].scatter(ds.rz[i], ds.rx[i])
    ax[1].set_title('z-x')
    ax[1].set_xlim(-7, 7)
    ax[1].set_ylim(-7, 7)

    fig.suptitle(f'{ds.time[i]:.2f} s')
    # plt.show()
plt.show()