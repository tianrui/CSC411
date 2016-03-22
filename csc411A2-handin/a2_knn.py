from a2_run_knn import *
from plot_digits import *
import matplotlib.pyplot as plt
from util import *
import numpy as np

def knn_script():
    train_in, valid_in, test_in, train_tar, valid_tar, test_tar = LoadData('digits.npz')
    valid_rate_vec = []
    test_rate_vec = []
    valid_tar = valid_tar.T
    test_tar = test_tar.T
    for i in [1, 3, 5, 7, 9]:
        c = 0
        predicted_labels = (run_knn(i, train_in, train_tar, valid_in))
        # print (predicted_labels.shape), 'valid_tar shape ', valid_tar.shape
        # raw_input()
        for j in range (0, len(predicted_labels)):
            # print predicted_labels[j], valid_tar[j]
            # raw_input()
            if predicted_labels[j] == valid_tar[j]:
                c = c + 1
        valid_rate_vec.append(1.0*c/len(predicted_labels))
        print 'Classification rate ', valid_rate_vec[-1]
        #print 'at k = ', i, 'c = ', c
    plt.plot([1, 3, 5, 7, 9], valid_rate_vec)
    plt.xlabel('K')
    plt.ylabel('Classification rate')
    plt.axis([0, 10, .70, 1.0])
    plt.show()
    raw_input()
    #if we choose k = 5, then k-2 and k+2 yield same class rates for validation
    #next is test run
    for i in [3, 5, 7]:
        c = 0
        predicted_labels = run_knn(i, train_in, train_tar, test_in)
        # print (predicted_labels.shape), 'valid_tar shape ', valid_tar.shape
        # raw_input()
        for j in range (0, len(predicted_labels)):
            if predicted_labels[j] == test_tar[j]:
                c = c + 1
        test_rate_vec.append(1.0*c/len(predicted_labels))
        print 'Classification rate ', test_rate_vec[-1]
        #print 'at k = ', i, 'c = ', c
    plt.plot([3, 5, 7], test_rate_vec)
    plt.xlabel('K')
    plt.ylabel('Classification rate')
    plt.axis([0, 10, .70, 1.0])
    plt.show()

if (__name__ == '__main__'):
    knn_script()
