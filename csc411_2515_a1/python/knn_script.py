from run_knn import *
from utils import *
from plot_digits import *
import matplotlib.pyplot as plt

train_in, train_tar = load_train()
valid_in, valid_tar = load_valid()
test_in, test_tar = load_test()
valid_rate_vec = []
test_rate_vec = []
for i in [1, 3, 5, 7, 9]:
    c = 0
    predicted_labels = run_knn(i, train_in, train_tar, valid_in)
    for j in range (0, len(predicted_labels)):
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

#if we choose k = 5, then k-2 and k+2 yield same class rates for validation
#next is test run
for i in [3, 5, 7]:
    c = 0
    predicted_labels = run_knn(i, train_in, train_tar, test_in)
    for j in range (0, len(predicted_labels)):
        if predicted_labels[j] == valid_tar[j]:
            c = c + 1
    test_rate_vec.append(1.0*c/len(predicted_labels))
    print 'Classification rate ', test_rate_vec[-1]
    #print 'at k = ', i, 'c = ', c
plt.plot([3, 5, 7], test_rate_vec)
plt.xlabel('K')
plt.ylabel('Classification rate')
plt.axis([0, 10, .70, 1.0])
plt.show()

print 'validation size:', len(valid_in), 'test size:', len(test_in)
#test rates were significantly higher than validation, could be due to 
#randomness in validation or test sets
