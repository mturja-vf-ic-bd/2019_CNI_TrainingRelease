from numpy import genfromtxt
from args import Args
from util import plot_signal, plot_matrix
import numpy as np


def parse_data(subject, parc):
    filename = Args.home + "/Training/" + subject + "/timeseries_" + parc + ".csv"
    data = genfromtxt(filename, delimiter=',')
    return data


if __name__ == '__main__':
    subject = 'sub-044'
    parc = 'aal'
    data = parse_data(subject, parc)
    print('data dim: {}'.format(parse_data(subject, parc).shape))

    # Plot first 6 row as signal
    #plot_signal(data[3:6, :])
    plot_matrix(np.dot(data.T, data))
    plot_matrix(np.dot(data, data.T))
