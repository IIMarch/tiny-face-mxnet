import scipy
import scipy.io
import numpy as np


def get_anchors():
    mat_f = scipy.io.loadmat('./RefBox_N25_scaled.mat')
    clusters = mat_f['clusters'][:,:4]

    shift_x = np.arange(0, 63) * 8
    shift_y = np.arange(0, 63) * 8
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    A = clusters.shape[0]
    K = shifts.shape[0]
    all_anchors = (clusters.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1,0,2)))
    all_anchors = all_anchors.reshape((K*A, 4))
    return all_anchors



