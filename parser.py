from numpy import genfromtxt
from args import Args
from util import plot_matrix, process_matrix, get_train_test_fold, plot_signal, plot_spectogram, get_spectogram
import numpy as np
from numpy.linalg import matrix_power
from numpy import corrcoef
import os
import csv


def parse_data(subject, parc):
    if parc == 'All':
        parc_all = ['aal', 'cc200', 'ho']
        data = []
        for parc in parc_all:
            filename = Args.home + "/Training/" + subject + "/timeseries_" + parc + ".csv"
            data.append(genfromtxt(filename, delimiter=','))
        data = np.concatenate(data, axis=0)
    else:
        filename = Args.home + "/Training/" + subject + "/timeseries_" + parc + ".csv"
        data = genfromtxt(filename, delimiter=',')

    return data


def get_class_ids():
    dict_class = {'ADHD': [], 'Control': [], 'All': []}
    label = []
    for sub in os.listdir(Args.data):
        filename = Args.home + "/Training/" + sub + "/phenotypic.csv"
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    s_id = row[0]
                    class_id = row[3]
                    dict_class[class_id].append(s_id)
                    dict_class['All'].append(s_id)
                    if class_id == 'Control':
                        label.append(0)
                    else:
                        label.append(1)
                line_count += 1

    return dict_class, label


def parse_data_class(class_id, parc):
    dict_class, label = get_class_ids()
    data = None
    for i, id in enumerate(dict_class[class_id]):
        single_data = parse_data(id, parc)
        if parc == 'aal' and single_data.shape != (116, 156) or \
                parc == 'ho' and single_data.shape != (112, 156) or \
                parc == 'cc200' and single_data.shape != (200, 156) or \
                parc == 'All' and single_data.shape != (428, 156):
            print("{} has bad shape {}".format(id, single_data.shape))
            continue
        print("{} shape: {}".format(id, single_data.shape))
        single_data -= single_data.mean(axis=1, keepdims=True)
        single_data /= single_data.std(axis=1, keepdims=True)
        single_data = get_spectogram(single_data)
        if i == 0:
            data = np.zeros((len(dict_class[class_id]), single_data.shape[0], single_data.shape[1], single_data.shape[2]))
        data[i] = single_data
    return np.array(data), np.array(label)


def load_data(parc):
    X, y = parse_data_class('All', parc)
    train_idx, test_idx = get_train_test_fold(X, y, 0.2)
    return X, y, train_idx, test_idx


if __name__ == '__main__':
    subject = 'sub-180'
    parc = 'All'
    data = parse_data(subject, parc)

    # time and space correlation
    # corr_time = np.dot(data.T, data)
    # corr_sp = np.dot(data, data.T)
    # corr_time = process_matrix(corrcoef(data.T), p=30)
    # corr_sp = process_matrix(corrcoef(data), p=90)

    # Plot first 6 row as signal
    # plot_signal(data[3:7, :])
    # plot_matrix(matrix_power(corr_time, 1))
    # plot_matrix(matrix_power(corr_sp, 1))
    # plot_matrix()
    # plot_matrix(matrix_power(corr_time, 4))
    # plot_matrix(matrix_power(corr_time, 8))
    # plot_matrix(np.dot(data, data.T))

    plot_spectogram(data[0, :])
