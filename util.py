from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import signal


def plot_signal(X):
    """
    :param X: N x F dimensional array. Contains N signal with F time point data.
    """

    plt_row, plt_col = X.shape
    for i in range(plt_row):
        plt.subplot(plt_row, 1, i + 1)
        plt.plot(X[i, :])
    plt.show()


def process_matrix(A, threshold=True, p=50):
    if threshold:
        percntl = np.percentile(A, p)
        A[A < percntl] = 0
        A[A >= percntl] = 1
    return A


def plot_matrix(A):
    plt.matshow(A)
    plt.show()


def get_train_test_fold(x, y, ratio=0.25):
    n_fold = int(1/ratio)
    train_fold = []
    test_fold = []
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    for train_index, test_index in kf.split(x, y):
        train_fold.append(train_index)
        test_fold.append(test_index)

    return train_fold, test_fold


def accuracy(output, y_true):
    output[output > 0.5] = 1
    output[output <=0.5] = 0
    return accuracy_score(y_true, output)


def get_spectogram(sig):
    fr = 1
    f, t, Sxx = signal.spectrogram(sig, fr, nperseg=5, noverlap=2)
    return f, t

def plot_spectogram(sig):
    fr = 1
    f, t, Sxx = signal.spectrogram(sig, fr, nperseg=8, noverlap=3)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
