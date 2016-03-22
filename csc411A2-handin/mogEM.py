from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary, init_mu, kmeans_iter=5):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  if init_mu == 'random':
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  elif init_mu == 'kmeans':
    mu = KMeans(x, K, kmeans_iter)
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])
    plotLogProb(logProbX[:i])

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def plotLogProb(logProbX):
    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(np.prod(logProbX.shape)), logProbX, 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()


def q2():
  iters = 20
  minVary = 0.01
  K = 2
  inputs_train2, inputs_valid2, inputs_test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', load2=True, load3=False)
  inputs_train3, inputs_valid3, inputs_test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', load2=False, load3=True)
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  p2, mu2, vary2, logProbX2 = mogEM(inputs_train2, K, iters, minVary, 'random')
  ShowMeans(mu2)
  ShowMeans(vary2)
  p3, mu3, vary3, logProbX2 = mogEM(inputs_train3, K, iters, minVary, 'random')
  ShowMeans(mu3)
  ShowMeans(vary3)
  raw_input('Press Enter to continue.')


def q3():
  iters = 10
  minVary = 0.01
  K = 20
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  pr, mur, varyr, logProbXr = mogEM(inputs_train, K, iters, minVary, 'random')
  raw_input('Press Enter to continue.')
  ShowMeans(mur)
  ShowMeans(varyr)
  
  pk, muk, varyk, logProbXk = mogEM(inputs_train, K, iters, minVary, 'kmeans')
  raw_input('Press Enter to continue.')
  ShowMeans(muk)
  ShowMeans(varyk)
  raw_input('Press Enter to continue.')

def q4():
  iters = 10
  minVary = 0.01
  test_labels = []
  valid_labels = []
  train_labels = []
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0]  
  errorTrain = np.zeros(T)
  errorTest = np.zeros(T)
  errorValidation = np.zeros(T)
  print(errorTrain)

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in xrange(T): 
    K = numComponents[t]
    valid_labels = []
    test_labels = []
    train_labels = []
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary, 'kmeans')
    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary, 'kmeans')
    
    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    T_valid, N_valid = inputs_valid.shape
    T_test, N_test = inputs_test.shape
    T_train, N_train = inputs_train.shape
    
    # classification on validation set
    moglog2 = mogLogProb(p2, mu2, vary2, inputs_valid)
    moglog3 = mogLogProb(p3, mu3, vary3, inputs_valid)
    prob_x = np.log(np.exp(moglog2) + np.exp(moglog3))
    mogprobd_2GivenX = moglog2 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x
    mogprobd_3GivenX = moglog3 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x
    for i in xrange(N_valid):
        if (mogprobd_2GivenX[i] > mogprobd_3GivenX[i]):
          valid_labels.append(0)
        else:
          valid_labels.append(1)
    class_err = np.count_nonzero(np.rint(np.abs(target_valid-valid_labels))) / (.01 * len(valid_labels))
    print class_err
    errorValidation[t] = class_err
    #raw_input()
    
    # test set
    moglog2 = mogLogProb(p2, mu2, vary2, inputs_test)
    moglog3 = mogLogProb(p3, mu3, vary3, inputs_test)
    prob_x = np.log(np.exp(moglog2) + np.exp(moglog3))
    mogprobd_2GivenX = moglog2 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x
    mogprobd_3GivenX = moglog3 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x
    for i in xrange(N_test):
      if (mogprobd_2GivenX[i] > mogprobd_3GivenX[i]):
        test_labels.append(0)
      else:
        test_labels.append(1)
    class_err = np.count_nonzero(np.rint(np.abs(target_test-test_labels))) / (.01 * len(test_labels))
    print class_err
    errorTest[t] = class_err
    #raw_input()
    
    # training set
    moglog2 = mogLogProb(p2, mu2, vary2, inputs_train)
    moglog3 = mogLogProb(p3, mu3, vary3, inputs_train)
    prob_x = np.log(np.exp(moglog2) + np.exp(moglog3))
    mogprobd_2GivenX = moglog2 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x
    mogprobd_3GivenX = moglog3 + \
                      (np.prod(train2.shape)/(np.prod(train2.shape) + np.prod(train3.shape))) - prob_x

    for i in xrange(N_train):
      if (mogprobd_2GivenX[i] > mogprobd_3GivenX[i]):
        train_labels.append(0)
      else:
        train_labels.append(1)
    class_err = np.count_nonzero(np.rint(np.abs(target_train-train_labels))) / (.01 * len(train_labels))
    print class_err
    errorTrain[t] = class_err
    #raw_input()
    
  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
  print errorTrain, errorValidation, errorTest
  plt.plot(numComponents, errorTrain, 'r-')
  plt.plot(numComponents, errorValidation, 'b-')
  plt.plot(numComponents, errorTest, 'y-')
  plt.title('Classification error rate vs. number of components')
  plt.xlabel('Number of components')
  plt.ylabel('Classification error rate(%)')
  plt.legend(['Train', 'Validation', 'Test'])
  plt.draw()
  raw_input('Press Enter to continue.')

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------
  K = 15
  iters = 10
  minVary = 0.01
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  # p2, mu2, vary2, logProbX2 = mogEM(inputs_train, K, iters, minVary, 'random')
  
  # p3, mu3, vary3, logProbX3 = mogEM(inputs_train, K, iters, minVary, 'kmeans')
 

  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  #q2()
  #q3()
  q4()
  #q5()

