from numpy import genfromtxt
from args import Args
from util import plot_signal, plot_matrix
import numpy as np
from numpy.linalg import matrix_power
from numpy import corrcoef


def parse_data(subject, parc):
    filename = Args.home + "/Training/" + subject + "/timeseries_" + parc + ".csv"
    data = genfromtxt(filename, delimiter=',')
    return data


if __name__ == '__main__':
    subject = 'sub-067'
    parc = 'aal'
    data = parse_data(subject, parc)
    print('data dim: {}'.format(parse_data(subject, parc).shape))

    ## time and space correlation
    # corr_time = np.dot(data.T, data)
    # corr_sp = np.dot(data, data.T)
    corr_time = corrcoef(data.T)
    corr_sp = corrcoef(data)

    # Plot first 6 row as signal
    #plot_signal(data[3:6, :])
    plot_matrix(matrix_power(corr_time, 16))
    plot_matrix(matrix_power(corr_sp, 1))
    # plot_matrix()
    # plot_matrix(matrix_power(corr_time, 4))
    # plot_matrix(matrix_power(corr_time, 8))
    #plot_matrix(np.dot(data, data.T))
