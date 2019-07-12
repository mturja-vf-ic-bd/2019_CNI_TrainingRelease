from matplotlib import pyplot as plt


def plot_signal(X):
    """
    :param X: N x F dimensional array. Contains N signal with F time point data.
    """

    plt_row, plt_col = X.shape
    for i in range(plt_row):
        plt.subplot(plt_row, 1, i + 1)
        plt.plot(X[i, :])
    plt.show()


def plot_matrix(A):
    plt.matshow(A)
    plt.show()
