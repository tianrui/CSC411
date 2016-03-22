from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def TrainNN(num_hiddens, eps, momentum, num_epochs):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  train_class_err = []
  valid_class_err = []
  num_train_cases = inputs_train.shape[1]
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))
    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    valid_CE, valid_class_err_epoch = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
    valid_class_err.append(valid_class_err_epoch)
    diff = np.abs(target_train-prediction)
    train_class_err_epoch = np.count_nonzero(np.rint(diff))/(.01*prediction.size)
    train_class_err.append(train_class_err_epoch)

    train_error.append(train_CE)
    valid_error.append(valid_CE)
    
    # Code for visualizing inputs to hiddens
    #if (epoch == int(num_epochs/10)):
        #print 'train_input shape', inputs_train.shape
        #print 'h_input shape', h_input.shape
        #ShowMeans(h_input)
        #raw_input()

    sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f' % (epoch, train_CE, valid_CE))
    sys.stdout.write('\n\rStep %d Train class err %.5f Validation class err %.5f' % (epoch, train_class_err_epoch, valid_class_err_epoch))
    sys.stdout.flush()
    # raw_input()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  sys.stdout.write('\n')
  final_train_error, final_train_class_err = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error, final_valid_class_err = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_error, final_test_class_err = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
  print 'Error: Train %.5f Validation %.5f Test %.5f' % (final_train_error, final_valid_error, final_test_error)
  print 'Class Error: Train %.5f Validation %.5f Test %.5f' % (final_train_class_err, final_valid_class_err, final_test_class_err)
  return W1, W2, b1, b2, train_error, valid_error, train_class_err, valid_class_err

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  class_err = np.count_nonzero(np.rint(np.abs(target-prediction))) / (.01 * prediction.size)
  return CE, class_err

def DisplayErrorPlot(train_error, valid_error):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel('Cross entropy')
  plt.legend()
  plt.draw()
  raw_input('Press Enter to exit.')

def DisplayClassErrorPlot(train_class_error, valid_class_error):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_class_error)), train_class_error, 'b', label='Train')
  plt.plot(range(len(valid_class_error)), valid_class_error, 'g', label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel('Classification error')
  plt.legend()
  plt.draw()
  raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def q2():
  num_hiddens = [10, 2, 5, 30, 100]
  eps = [0.1, 0.01, 0.2, 0.5, 0.02]
  momentum = [0.0, 0.5, 0.9]
  num_epochs = 500
  # Load model if necessary
  outputfile = 'model.npz'
  # W1, W2, b1, b2, train_error, valid_error = LoadModel(outputfile);
  for hiddens_i in num_hiddens[1:2]:
    W1, W2, b1, b2, train_error, valid_error, train_class_err, valid_class_err = TrainNN(hiddens_i, eps[-1], momentum[1], num_epochs)
    DisplayErrorPlot(train_error, valid_error)
    DisplayClassErrorPlot(train_class_err, valid_class_err)
  # If you wish to save the model for future use :
  # SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

def q3_5():
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  num_hiddens = 30
  eps = 0.1
  momentum= 0.5
  num_epochs = 200
  W1, W2, b1, b2, train_error, valid_error, train_class_err, valid_class_err = TrainNN(num_hiddens, eps, momentum, num_epochs)
  DisplayErrorPlot(train_error, valid_error)
  DisplayClassErrorPlot(train_class_err, valid_class_err)

def ShowMeans(means):
  """Show the cluster centers as images."""
  print 'input shape', means.shape
  raw_input()
  plt.figure(1)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')



if __name__ == '__main__':
  q2()
  #q3_5()
